#!/usr/bin/env python3
"""
Smart Component Placer for Atopile Modules

Parses atopile module definitions to extract design rules and placement
constraints from:
1. Module docstrings (layout guidelines)
2. Component connections (topology)
3. Component types and packages

Uses these rules to arrange components optimally for routing.

Usage:
    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3 smart_placer.py [--modules all|module1,module2]
"""

import os
import sys
import re
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False
    logger.warning("pcbnew not available - run with KiCad's Python")

# Optional: simulated annealing support
try:
    from simanneal import Annealer
    SIMANNEAL_AVAILABLE = True
except ImportError:
    SIMANNEAL_AVAILABLE = False
    Annealer = None  # Placeholder for type hints
    logger.debug("simanneal not available - install with: pip install simanneal")

import random  # For simulated annealing


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ModuleTopology(Enum):
    """Module topology types affecting placement strategy"""
    IC_CENTRIC = "ic_centric"      # IC with decoupling caps (sensors)
    POWER_CHAIN = "power_chain"    # Sequential power path (power filter)
    RF_LINEAR = "rf_linear"        # Linear RF signal path (matching network)
    BRIDGE = "bridge"              # Simple 2-component bridge (I2C pullups)
    GENERIC = "generic"            # Default grid placement


class ComponentType(Enum):
    """Component types for placement priority"""
    IC = "ic"                      # Main IC - center placement
    DECOUPLING_HF = "decoupling_hf"  # High-freq decoupling - closest to IC
    DECOUPLING_BULK = "decoupling_bulk"  # Bulk decoupling - near IC
    INDUCTOR = "inductor"          # Ferrites/inductors
    RF_COMPONENT = "rf"            # RF path components
    RESISTOR = "resistor"          # Generic resistors
    CAPACITOR = "capacitor"        # Generic capacitors
    CONNECTOR = "connector"        # Connectors, antennas
    PASSIVE = "passive"            # Other passives


@dataclass
class PlacementRule:
    """Extracted placement rule from docstring"""
    component_pattern: str  # Regex pattern for component name
    rule_type: str          # "proximity", "order", "position"
    target: str            # What it's relative to (e.g., "IC", "VDD")
    distance_mm: float = 0.0  # Distance constraint in mm
    priority: int = 0      # Placement priority (lower = place first)


@dataclass
class ComponentInfo:
    """Component information extracted from .ato file"""
    name: str              # atopile name (e.g., c_hf, ic)
    ref: str = ""          # KiCad reference (e.g., C1, U1)
    comp_type: ComponentType = ComponentType.PASSIVE
    package: str = ""
    connections: List[str] = field(default_factory=list)
    placement_distance: float = 5.0  # Default distance from center
    placement_angle: float = 0.0     # Angle from center (radians)
    placement_order: int = 0         # Sequential placement order
    width_mm: float = 1.5
    height_mm: float = 1.0


@dataclass
class ModuleDesignRules:
    """Complete design rules for a module"""
    module_name: str
    topology: ModuleTopology
    components: Dict[str, ComponentInfo]
    placement_rules: List[PlacementRule]
    has_ic: bool = False
    rf_path: List[str] = field(default_factory=list)
    power_path: List[str] = field(default_factory=list)


# ============================================================================
# SIMULATED ANNEALING PLACER
# ============================================================================

class PCBPlacementAnnealer:
    """Simulated annealing optimizer for PCB component placement

    This class implements the simanneal interface to optimize component
    positions by minimizing total wire length while avoiding overlaps.

    If simanneal is not installed, this class will not be functional.
    """

    def __init__(self, positions: Dict[str, Tuple[float, float]],
                 netlist_analyzer, center: Tuple[float, float],
                 min_distance: float = 2.0):
        """Initialize the annealer

        Args:
            positions: Initial positions dict {ref: (x, y)}
            netlist_analyzer: NetlistAnalyzer instance for wire length calculation
            center: Center point for bounds
            min_distance: Minimum distance between components (mm)
        """
        self.netlist_analyzer = netlist_analyzer
        self.center = center
        self.min_distance = min_distance
        self.refs = list(positions.keys())

        # Convert positions dict to state (list of coordinates for mutability)
        self.state = positions.copy()

        # Annealing parameters (can be overridden)
        self.Tmax = 25.0
        self.Tmin = 0.1
        self.steps = 10000
        self.updates = 0

    def move(self):
        """Make a random move: adjust one component's position"""
        # Pick a random component
        ref = random.choice(self.refs)
        x, y = self.state[ref]

        # Random displacement (larger at high temperature)
        # Temperature is handled by simanneal, so use fixed range
        dx = random.uniform(-1.5, 1.5)
        dy = random.uniform(-1.5, 1.5)

        new_x = x + dx
        new_y = y + dy

        # Keep within bounds
        new_x = max(self.center[0] - 20, min(self.center[0] + 20, new_x))
        new_y = max(self.center[1] - 20, min(self.center[1] + 20, new_y))

        self.state[ref] = (new_x, new_y)

    def energy(self) -> float:
        """Calculate energy (cost) of current state

        Lower energy = better placement
        Energy = wire_length + overlap_penalty
        """
        # Wire length component
        wire_length = self.netlist_analyzer.calculate_wire_length(self.state)

        # Overlap penalty component
        overlap_penalty = 0.0
        refs_list = list(self.state.keys())

        for i, ref1 in enumerate(refs_list):
            x1, y1 = self.state[ref1]
            for ref2 in refs_list[i+1:]:
                x2, y2 = self.state[ref2]

                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if dist < self.min_distance:
                    # Strong penalty for overlap
                    overlap_penalty += (self.min_distance - dist) ** 2 * 100

        return wire_length + overlap_penalty

    def anneal(self) -> Tuple[Dict[str, Tuple[float, float]], float]:
        """Run simulated annealing optimization

        Returns:
            Tuple of (final_positions, final_energy)
        """
        if not SIMANNEAL_AVAILABLE:
            return self.state, self.energy()

        # Create actual Annealer subclass instance
        class _Annealer(Annealer):
            def __init__(inner_self, state, parent):
                inner_self.parent = parent
                super().__init__(state)

            def move(inner_self):
                # Pick a random component
                ref = random.choice(inner_self.parent.refs)
                x, y = inner_self.state[ref]

                dx = random.uniform(-1.5, 1.5)
                dy = random.uniform(-1.5, 1.5)

                new_x = max(inner_self.parent.center[0] - 20,
                           min(inner_self.parent.center[0] + 20, x + dx))
                new_y = max(inner_self.parent.center[1] - 20,
                           min(inner_self.parent.center[1] + 20, y + dy))

                inner_self.state[ref] = (new_x, new_y)

            def energy(inner_self) -> float:
                wire_length = inner_self.parent.netlist_analyzer.calculate_wire_length(inner_self.state)

                overlap_penalty = 0.0
                refs_list = list(inner_self.state.keys())

                for i, ref1 in enumerate(refs_list):
                    x1, y1 = inner_self.state[ref1]
                    for ref2 in refs_list[i+1:]:
                        x2, y2 = inner_self.state[ref2]

                        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if dist < inner_self.parent.min_distance:
                            overlap_penalty += (inner_self.parent.min_distance - dist) ** 2 * 100

                return wire_length + overlap_penalty

        annealer = _Annealer(self.state.copy(), self)
        annealer.Tmax = self.Tmax
        annealer.Tmin = self.Tmin
        annealer.steps = self.steps
        annealer.updates = self.updates

        final_state, final_energy = annealer.anneal()
        return final_state, final_energy


# ============================================================================
# ATO FILE PARSER
# ============================================================================

class AtoModuleParser:
    """Parses atopile module files to extract design rules"""

    # Patterns for extracting information
    PATTERNS = {
        'module_def': re.compile(r'module\s+(\w+):'),
        'component_new': re.compile(r'(\w+)\s*=\s*new\s+(\w+)'),
        'package': re.compile(r'(\w+)\.package\s*=\s*"([^"]+)"'),
        'capacitance': re.compile(r'(\w+)\.capacitance\s*=\s*([\d.]+)([munpf]?F)'),
        'inductance': re.compile(r'(\w+)\.inductance\s*=\s*([\d.]+)([munp]?H)'),
        'connection': re.compile(r'(\w+(?:\.\w+)*)\s*~\s*(\w+(?:\.\w+)*)'),
        'bridge': re.compile(r'(\w+(?:\.\w+)*)\s*~>\s*(\w+)\s*~>\s*(\w+(?:\.\w+)*)'),
        # Docstring patterns for layout rules
        'within_mm': re.compile(r'[Ww]ithin\s*([\d.]+)[-–]?([\d.]*)\s*mm\s+of\s+(\w+)'),
        'closest_to': re.compile(r'[Cc]losest?\s+to\s+(\w+)'),
        'linear_chain': re.compile(r'[Ll]inear\s+chain'),
        'power_path': re.compile(r'[Pp]ower\s+path|[Rr]oute.*→'),
    }

    # Component type detection
    COMPONENT_TYPES = {
        'Capacitor': ComponentType.CAPACITOR,
        'Resistor': ComponentType.RESISTOR,
        'Inductor': ComponentType.INDUCTOR,
        'FerriteBead': ComponentType.INDUCTOR,
        'FerriteBead_120ohm': ComponentType.INDUCTOR,
        'Antenna': ComponentType.CONNECTOR,
    }

    # Package sizes (approximate, in mm)
    PACKAGE_SIZES = {
        '0201': (0.6, 0.3),
        '0402': (1.0, 0.5),
        '0603': (1.6, 0.8),
        '0805': (2.0, 1.25),
        '1206': (3.2, 1.6),
    }

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.modules_dir = os.path.join(project_dir, "elec", "modules")

    def parse_module(self, module_name: str) -> Optional[ModuleDesignRules]:
        """Parse a single module's .ato file"""
        # Find the .ato file
        ato_file = self._find_ato_file(module_name)
        if not ato_file:
            logger.warning(f"No .ato file found for module: {module_name}")
            return None

        with open(ato_file, 'r') as f:
            content = f.read()

        # Extract docstring
        docstring = self._extract_docstring(content)

        # Detect topology from docstring
        topology = self._detect_topology(docstring, content)

        # Extract components
        components = self._extract_components(content)

        # Extract connections
        self._extract_connections(content, components)

        # Extract placement rules from docstring
        placement_rules = self._extract_placement_rules(docstring)

        # Classify components based on context
        self._classify_components(components, content)

        # Detect if module has IC
        has_ic = any(c.comp_type == ComponentType.IC for c in components.values())

        # Extract RF/power paths if present
        rf_path = self._extract_rf_path(content, components) if topology == ModuleTopology.RF_LINEAR else []
        power_path = self._extract_power_path(content, components) if topology == ModuleTopology.POWER_CHAIN else []

        return ModuleDesignRules(
            module_name=module_name,
            topology=topology,
            components=components,
            placement_rules=placement_rules,
            has_ic=has_ic,
            rf_path=rf_path,
            power_path=power_path
        )

    def _find_ato_file(self, module_name: str) -> Optional[str]:
        """Find the .ato file for a module"""
        # Map module names to file names
        name_mappings = {
            'power_filter': 'power_filter_module.ato',
            'i2c_bus': 'i2c_bus_module.ato',
            'lis3dhtr': 'lis3dhtr_module.ato',
            'qmi8658c': 'qmi8658c_module.ato',
            'hdc2080': 'hdc2080_module.ato',
            'eeprom': 'eeprom_module.ato',
            'rf_matching': 'rf_matching_module.ato',
            'status_led': 'status_led_module.ato',
            'ntc_temp': 'ntc_temp_module.ato',
            'swd_debug': 'swd_debug_module.ato',
        }

        filename = name_mappings.get(module_name, f"{module_name}_module.ato")
        filepath = os.path.join(self.modules_dir, filename)

        if os.path.exists(filepath):
            return filepath
        return None

    def _extract_docstring(self, content: str) -> str:
        """Extract the module docstring"""
        # Find triple-quoted docstrings
        match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def _detect_topology(self, docstring: str, content: str) -> ModuleTopology:
        """Detect module topology from docstring and content"""
        docstring_lower = docstring.lower()

        if 'rf' in docstring_lower and 'linear chain' in docstring_lower:
            return ModuleTopology.RF_LINEAR

        if 'power path' in docstring_lower or 'power distribution' in docstring_lower:
            return ModuleTopology.POWER_CHAIN

        # Check for IC presence
        if re.search(r'ic\s*=\s*new\s+\w+', content):
            return ModuleTopology.IC_CENTRIC

        # Check for simple bridge (2 components)
        components = self.PATTERNS['component_new'].findall(content)
        passive_count = sum(1 for _, ctype in components
                          if ctype in ['Resistor', 'Capacitor', 'Inductor'])
        if passive_count == 2 and len(components) == 2:
            return ModuleTopology.BRIDGE

        return ModuleTopology.GENERIC

    def _extract_components(self, content: str) -> Dict[str, ComponentInfo]:
        """Extract component declarations"""
        components = {}

        for match in self.PATTERNS['component_new'].finditer(content):
            name = match.group(1)
            comp_class = match.group(2)

            # Skip interfaces (ElectricPower, I2C, etc.)
            if comp_class in ['ElectricPower', 'I2C', 'ElectricLogic', 'Electrical', 'SPI', 'SWD']:
                continue

            comp_type = self.COMPONENT_TYPES.get(comp_class, ComponentType.PASSIVE)

            # Check if it's an IC (usually named 'ic' or starts with uppercase)
            if name == 'ic' or (comp_class[0].isupper() and comp_class not in self.COMPONENT_TYPES):
                comp_type = ComponentType.IC

            components[name] = ComponentInfo(
                name=name,
                comp_type=comp_type
            )

        # Extract packages
        for match in self.PATTERNS['package'].finditer(content):
            comp_name = match.group(1)
            package = match.group(2)
            if comp_name in components:
                components[comp_name].package = package
                if package in self.PACKAGE_SIZES:
                    w, h = self.PACKAGE_SIZES[package]
                    components[comp_name].width_mm = w + 0.5  # Add clearance
                    components[comp_name].height_mm = h + 0.5

        return components

    def _extract_connections(self, content: str, components: Dict[str, ComponentInfo]):
        """Extract component connections"""
        # Regular connections
        for match in self.PATTERNS['connection'].finditer(content):
            src = match.group(1).split('.')[0]
            dst = match.group(2).split('.')[0]

            if src in components:
                components[src].connections.append(dst)
            if dst in components:
                components[dst].connections.append(src)

        # Bridge connections
        for match in self.PATTERNS['bridge'].finditer(content):
            src = match.group(1).split('.')[0]
            bridge = match.group(2)
            dst = match.group(3).split('.')[0]

            if bridge in components:
                components[bridge].connections.extend([src, dst])

    def _classify_components(self, components: Dict[str, ComponentInfo], content: str):
        """Classify components based on their context and naming"""
        for name, comp in components.items():
            # Classify capacitors by their context
            if comp.comp_type == ComponentType.CAPACITOR:
                name_lower = name.lower()

                # Check capacitance value
                cap_match = self.PATTERNS['capacitance'].search(content)
                if cap_match and cap_match.group(1) == name:
                    value = float(cap_match.group(2))
                    unit = cap_match.group(3)

                    # Normalize to Farads
                    multipliers = {'F': 1, 'mF': 1e-3, 'uF': 1e-6, 'nF': 1e-9, 'pF': 1e-12}
                    cap_value = value * multipliers.get(unit, 1e-9)

                    # Small caps (100nF or less) are usually HF decoupling
                    if cap_value <= 100e-9:
                        if 'hf' in name_lower or 'bypass' in name_lower or 'vdd' in name_lower:
                            comp.comp_type = ComponentType.DECOUPLING_HF
                            comp.placement_distance = 1.5
                        else:
                            comp.comp_type = ComponentType.DECOUPLING_HF
                            comp.placement_distance = 2.0
                    else:
                        # Larger caps are bulk decoupling
                        comp.comp_type = ComponentType.DECOUPLING_BULK
                        comp.placement_distance = 3.0

                # Naming patterns
                if 'hf' in name_lower or 'bypass' in name_lower:
                    comp.comp_type = ComponentType.DECOUPLING_HF
                    comp.placement_distance = 1.5
                elif 'bulk' in name_lower:
                    comp.comp_type = ComponentType.DECOUPLING_BULK
                    comp.placement_distance = 3.0
                elif 'rf' in name_lower or 'match' in name_lower or 'harmonic' in name_lower:
                    comp.comp_type = ComponentType.RF_COMPONENT
                    comp.placement_distance = 2.0

            # Classify inductors
            elif comp.comp_type == ComponentType.INDUCTOR:
                name_lower = name.lower()
                if 'harmonic' in name_lower or 'series' in name_lower:
                    comp.comp_type = ComponentType.RF_COMPONENT

    def _extract_placement_rules(self, docstring: str) -> List[PlacementRule]:
        """Extract placement rules from docstring"""
        rules = []

        # "within X-Y mm of VDD"
        for match in self.PATTERNS['within_mm'].finditer(docstring):
            min_dist = float(match.group(1))
            max_dist = float(match.group(2)) if match.group(2) else min_dist
            target = match.group(3)

            rules.append(PlacementRule(
                component_pattern=".*",
                rule_type="proximity",
                target=target,
                distance_mm=(min_dist + max_dist) / 2,
                priority=1
            ))

        # "closest to IC"
        for match in self.PATTERNS['closest_to'].finditer(docstring):
            target = match.group(1)
            rules.append(PlacementRule(
                component_pattern=".*",
                rule_type="closest",
                target=target,
                distance_mm=1.0,
                priority=0
            ))

        return rules

    def _extract_rf_path(self, content: str, components: Dict[str, ComponentInfo]) -> List[str]:
        """Extract RF signal path order"""
        # Look for sequential connections in RF modules
        path = []
        # This would require more sophisticated parsing
        # For now, return components in a logical order
        for name, comp in components.items():
            if comp.comp_type == ComponentType.RF_COMPONENT:
                path.append(name)
        return path

    def _extract_power_path(self, content: str, components: Dict[str, ComponentInfo]) -> List[str]:
        """Extract power distribution path order"""
        path = []
        for name, comp in components.items():
            if comp.comp_type in [ComponentType.INDUCTOR, ComponentType.DECOUPLING_BULK]:
                path.append(name)
        return path


# ============================================================================
# NETLIST ANALYZER
# ============================================================================

@dataclass
class PadInfo:
    """Information about a pad/pin"""
    ref: str           # Component reference (e.g., "R1")
    pad_name: str      # Pad name/number (e.g., "1")
    net_name: str      # Net name (e.g., "VCC")
    position: Tuple[float, float]  # Position in mm


@dataclass
class ComponentConnectivity:
    """Connectivity information for a component"""
    ref: str
    pads: List[PadInfo]
    connected_refs: Set[str]  # Other components connected to this one
    net_names: Set[str]       # Nets this component is part of
    connection_weights: Dict[str, int]  # ref -> number of shared nets


class NetlistAnalyzer:
    """Analyzes PCB netlist for component connectivity"""

    def __init__(self, board):
        self.board = board
        self.nets: Dict[str, List[PadInfo]] = {}  # net_name -> pads
        self.component_connectivity: Dict[str, ComponentConnectivity] = {}
        self._analyze()

    def _analyze(self):
        """Analyze board netlist"""
        if not self.board:
            return

        # Build net to pads mapping
        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            pos = fp.GetPosition()

            for pad in fp.Pads():
                net = pad.GetNet()
                if net:
                    net_name = net.GetNetname()
                    if net_name:
                        pad_pos = pad.GetPosition()
                        pad_info = PadInfo(
                            ref=ref,
                            pad_name=pad.GetNumber(),
                            net_name=net_name,
                            position=(pcbnew.ToMM(pad_pos.x), pcbnew.ToMM(pad_pos.y))
                        )

                        if net_name not in self.nets:
                            self.nets[net_name] = []
                        self.nets[net_name].append(pad_info)

        # Build component connectivity from nets
        self._build_connectivity()

    def _build_connectivity(self):
        """Build component connectivity graph from nets"""
        # Initialize connectivity for all components
        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            self.component_connectivity[ref] = ComponentConnectivity(
                ref=ref,
                pads=[],
                connected_refs=set(),
                net_names=set(),
                connection_weights={}
            )

        # Populate from nets
        for net_name, pads in self.nets.items():
            # Get all refs connected by this net
            refs_in_net = set(p.ref for p in pads)

            for pad in pads:
                if pad.ref in self.component_connectivity:
                    conn = self.component_connectivity[pad.ref]
                    conn.pads.append(pad)
                    conn.net_names.add(net_name)

                    # Add connections to other components
                    for other_ref in refs_in_net:
                        if other_ref != pad.ref:
                            conn.connected_refs.add(other_ref)
                            conn.connection_weights[other_ref] = \
                                conn.connection_weights.get(other_ref, 0) + 1

    def get_connected_components(self, ref: str) -> Dict[str, int]:
        """Get components connected to ref with connection weight

        Returns:
            Dict mapping ref -> number of shared connections
        """
        if ref in self.component_connectivity:
            return self.component_connectivity[ref].connection_weights
        return {}

    def get_strongest_connection(self, ref: str) -> Optional[str]:
        """Get the component most strongly connected to ref"""
        connections = self.get_connected_components(ref)
        if connections:
            return max(connections, key=connections.get)
        return None

    def get_optimal_placement_order(self) -> List[str]:
        """Get order for placing components to minimize wire length

        Uses a greedy algorithm starting from the most connected component.
        """
        if not self.component_connectivity:
            return []

        # Find most connected component (highest total weight)
        def total_connections(ref):
            conn = self.component_connectivity.get(ref)
            if conn:
                return sum(conn.connection_weights.values())
            return 0

        # Start with most connected
        all_refs = list(self.component_connectivity.keys())
        all_refs.sort(key=total_connections, reverse=True)

        if not all_refs:
            return []

        ordered = [all_refs[0]]
        remaining = set(all_refs[1:])

        # Greedily add next most connected component
        while remaining:
            best_next = None
            best_score = -1

            for candidate in remaining:
                # Score = sum of connections to already-placed components
                conn = self.component_connectivity.get(candidate)
                if conn:
                    score = sum(conn.connection_weights.get(placed, 0)
                               for placed in ordered)
                    if score > best_score:
                        best_score = score
                        best_next = candidate

            if best_next:
                ordered.append(best_next)
                remaining.remove(best_next)
            elif remaining:
                # No connections, just add the first remaining
                ordered.append(remaining.pop())

        return ordered

    def calculate_wire_length(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """Calculate total estimated wire length for given positions

        Uses Manhattan distance between connected pads.
        """
        total = 0.0

        for net_name, pads in self.nets.items():
            if len(pads) < 2:
                continue

            # Get positions for this net's pads
            pad_positions = []
            for pad in pads:
                if pad.ref in positions:
                    # Use component center position
                    pad_positions.append(positions[pad.ref])

            if len(pad_positions) < 2:
                continue

            # Calculate minimum spanning tree length (approximation)
            # For simplicity, use sum of distances to centroid
            cx = sum(p[0] for p in pad_positions) / len(pad_positions)
            cy = sum(p[1] for p in pad_positions) / len(pad_positions)

            for px, py in pad_positions:
                total += abs(px - cx) + abs(py - cy)

        return total


# ============================================================================
# PCB COMPONENT MAPPER
# ============================================================================

class PCBComponentMapper:
    """Maps atopile component names to KiCad reference designators"""

    def __init__(self, pcb_path: str):
        self.pcb_path = pcb_path
        self.board = None
        self.ato_to_ref: Dict[str, str] = {}
        self.ref_to_footprint: Dict[str, any] = {}
        self.netlist_analyzer: Optional[NetlistAnalyzer] = None

        if PCBNEW_AVAILABLE and os.path.exists(pcb_path):
            try:
                self.board = pcbnew.LoadBoard(pcb_path)
                self._build_mapping()
                self.netlist_analyzer = NetlistAnalyzer(self.board)
            except Exception as e:
                logger.warning(f"Failed to load PCB: {e}")

    def _build_mapping(self):
        """Build mapping between atopile addresses and KiCad refs"""
        if not self.board:
            return

        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            self.ref_to_footprint[ref] = fp

            # Get atopile_address from field (KiCad 9 API)
            ato_addr = None
            try:
                if fp.HasFieldByName('atopile_address'):
                    field = fp.GetFieldByName('atopile_address')
                    ato_addr = field.GetText()
            except Exception as e:
                logger.debug(f"Could not get atopile_address for {ref}: {e}")

            if ato_addr:
                self.ato_to_ref[ato_addr] = ref
                logger.debug(f"  Mapped: {ato_addr} -> {ref}")

    def get_ref_for_ato_name(self, ato_name: str) -> Optional[str]:
        """Get KiCad reference for atopile component name"""
        return self.ato_to_ref.get(ato_name)

    def get_footprint(self, ref: str):
        """Get footprint object by reference"""
        return self.ref_to_footprint.get(ref)

    def get_all_refs(self) -> List[str]:
        """Get all component references"""
        return list(self.ref_to_footprint.keys())


# ============================================================================
# PLACEMENT MANAGER (Collision Detection)
# ============================================================================

@dataclass
class BoundingBox:
    """Axis-aligned bounding box for collision detection"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def overlaps(self, other: 'BoundingBox', margin: float = 0.3) -> bool:
        """Check if this box overlaps with another (with margin in mm)"""
        return not (
            self.x_max + margin < other.x_min or
            self.x_min - margin > other.x_max or
            self.y_max + margin < other.y_min or
            self.y_min - margin > other.y_max
        )

    def center(self) -> Tuple[float, float]:
        """Get center point"""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def width(self) -> float:
        return self.x_max - self.x_min

    def height(self) -> float:
        return self.y_max - self.y_min


class PlacementManager:
    """Manages component placement with collision detection"""

    def __init__(self, board):
        self.board = board
        self.placed_boxes: Dict[str, BoundingBox] = {}

    def get_footprint_bbox(self, fp) -> BoundingBox:
        """Get bounding box of a footprint in mm"""
        try:
            # Get position and estimate size from pads
            pos = fp.GetPosition()
            x = pcbnew.ToMM(pos.x)
            y = pcbnew.ToMM(pos.y)

            # Get bounding box from pads
            pads_bbox = fp.GetFpPadsLocalBbox()
            width = max(pcbnew.ToMM(pads_bbox.GetWidth()), 1.0) + 0.5
            height = max(pcbnew.ToMM(pads_bbox.GetHeight()), 1.0) + 0.5

            return BoundingBox(
                x_min=x - width / 2,
                y_min=y - height / 2,
                x_max=x + width / 2,
                y_max=y + height / 2
            )
        except Exception as e:
            # Fallback to position with default size
            logger.debug(f"    Could not get bbox, using default: {e}")
            pos = fp.GetPosition()
            x = pcbnew.ToMM(pos.x)
            y = pcbnew.ToMM(pos.y)
            return BoundingBox(
                x_min=x - 1.0,
                y_min=y - 0.5,
                x_max=x + 1.0,
                y_max=y + 0.5
            )

    def get_bbox_at_position(self, fp, x: float, y: float) -> BoundingBox:
        """Get bounding box if footprint were at given position"""
        current_bbox = self.get_footprint_bbox(fp)
        current_center = current_bbox.center()

        # Calculate offset
        dx = x - current_center[0]
        dy = y - current_center[1]

        return BoundingBox(
            x_min=current_bbox.x_min + dx,
            y_min=current_bbox.y_min + dy,
            x_max=current_bbox.x_max + dx,
            y_max=current_bbox.y_max + dy
        )

    def check_overlap(self, ref: str, bbox: BoundingBox, margin: float = 0.3) -> bool:
        """Check if a bounding box overlaps with any placed component"""
        for placed_ref, placed_bbox in self.placed_boxes.items():
            if placed_ref != ref and bbox.overlaps(placed_bbox, margin):
                return True
        return False

    def find_non_overlapping_position(self, fp, ref: str, target_x: float, target_y: float,
                                       max_attempts: int = 36, search_radius: float = 2.0) -> Tuple[float, float]:
        """Find a non-overlapping position near the target

        Searches in a spiral pattern around the target position.
        """
        bbox = self.get_bbox_at_position(fp, target_x, target_y)

        # Check if target position is clear
        if not self.check_overlap(ref, bbox):
            return target_x, target_y

        # Spiral search for clear position
        for radius_mult in range(1, 6):  # Increase search radius
            radius = search_radius * radius_mult
            for i in range(max_attempts):
                angle = (2 * math.pi * i) / max_attempts
                x = target_x + radius * math.cos(angle)
                y = target_y + radius * math.sin(angle)

                bbox = self.get_bbox_at_position(fp, x, y)
                if not self.check_overlap(ref, bbox):
                    logger.debug(f"    Found clear position at ({x:.1f}, {y:.1f}) after offset")
                    return x, y

        # Fallback: return target position with warning
        logger.warning(f"    Could not find non-overlapping position for {ref}")
        return target_x, target_y

    def register_placement(self, ref: str, fp):
        """Register a placed component's bounding box"""
        bbox = self.get_footprint_bbox(fp)
        self.placed_boxes[ref] = bbox

    def clear_existing_traces(self):
        """Delete all existing traces and vias from the board"""
        if not self.board:
            return 0

        deleted = 0

        # Collect tracks to delete (can't modify while iterating)
        tracks_to_delete = []
        for track in self.board.GetTracks():
            tracks_to_delete.append(track)

        # Delete tracks
        for track in tracks_to_delete:
            self.board.Remove(track)
            deleted += 1

        logger.info(f"  Deleted {deleted} existing traces/vias")
        return deleted


# ============================================================================
# SMART PLACER
# ============================================================================

class SmartPlacer:
    """Intelligent component placement using design rules"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.parser = AtoModuleParser(project_dir)
        self.placement_mgr: Optional[PlacementManager] = None

    def place_module(self, module_name: str, center: Tuple[float, float] = (100.0, 100.0),
                     use_netlist: bool = True, refine_method: str = "anneal") -> int:
        """Place components in a module PCB according to design rules

        Args:
            module_name: Name of the module to place
            center: Center point for placement (mm)
            use_netlist: Use netlist connectivity for optimization
            refine_method: Refinement method - "force" (force-directed),
                          "anneal" (simulated annealing), or "none"

        Returns:
            Number of components placed
        """
        if not PCBNEW_AVAILABLE:
            logger.error("pcbnew required for placement")
            return 0

        # Parse module design rules
        design_rules = self.parser.parse_module(module_name)
        if not design_rules:
            logger.warning(f"Could not parse design rules for {module_name}")
            return self._fallback_placement(module_name, center)

        # Load PCB
        pcb_path = os.path.join(
            self.project_dir, "layouts", module_name, f"{module_name}.kicad_pcb"
        )
        mapper = PCBComponentMapper(pcb_path)

        if not mapper.board:
            logger.error(f"Could not load PCB: {pcb_path}")
            return 0

        # Initialize placement manager with collision detection
        self.placement_mgr = PlacementManager(mapper.board)

        logger.info(f"Placing {module_name} with topology: {design_rules.topology.value}")

        # Log netlist info if available
        if mapper.netlist_analyzer and use_netlist:
            nets = len(mapper.netlist_analyzer.nets)
            logger.info(f"  Netlist: {nets} nets, optimizing for shortest routes")

        # Choose placement strategy based on topology and netlist
        if use_netlist and design_rules.topology == ModuleTopology.GENERIC:
            # For generic topology, use pure netlist-aware placement
            placed = self._place_netlist_aware(mapper, design_rules, center)
        elif design_rules.topology == ModuleTopology.IC_CENTRIC:
            placed = self._place_ic_centric_netlist(mapper, design_rules, center, use_netlist)
        elif design_rules.topology == ModuleTopology.RF_LINEAR:
            placed = self._place_rf_linear(mapper, design_rules, center)
        elif design_rules.topology == ModuleTopology.POWER_CHAIN:
            placed = self._place_power_chain(mapper, design_rules, center)
        elif design_rules.topology == ModuleTopology.BRIDGE:
            placed = self._place_bridge_netlist(mapper, design_rules, center, use_netlist)
        else:
            placed = self._place_netlist_aware(mapper, design_rules, center)

        # Run placement refinement to optimize placement
        if use_netlist and placed > 1 and refine_method != "none":
            if refine_method == "anneal":
                self._refine_with_annealing(mapper, center)
            else:  # Default to force-directed
                self._refine_placement(mapper, center)

        # Adjust silkscreen text to avoid overlapping components
        self._adjust_silkscreen_text(mapper)

        # Save PCB
        mapper.board.Save(pcb_path)
        logger.info(f"  Placed {placed} components, saved to {pcb_path}")

        return placed

    def _place_ic_centric_netlist(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                                   center: Tuple[float, float], use_netlist: bool = True) -> int:
        """Place components radially around IC with netlist optimization

        Uses IC as center, then places connected components based on:
        1. Which IC pins they connect to
        2. Connection weight (more connections = closer)
        """
        placed = 0
        analyzer = mapper.netlist_analyzer if use_netlist else None
        positions: Dict[str, Tuple[float, float]] = {}

        # Find and place IC at center
        ic_ref = None
        for name, comp in rules.components.items():
            if comp.comp_type == ComponentType.IC:
                ref = mapper.get_ref_for_ato_name(name)
                if ref:
                    fp = mapper.get_footprint(ref)
                    if fp:
                        x, y = self._set_position_safe(fp, ref, center[0], center[1])
                        positions[ref] = (x, y)
                        ic_ref = ref
                        placed += 1
                        logger.debug(f"  Placed IC {ref} at center")
                break

        if not ic_ref:
            # No IC found, use netlist-aware placement
            return self._place_netlist_aware(mapper, rules, center)

        # Get components connected to IC sorted by connection weight
        refs_to_place = []
        if analyzer:
            ic_connections = analyzer.get_connected_components(ic_ref)
            # Sort by connection weight (more connections = place first/closer)
            refs_to_place = sorted(ic_connections.keys(),
                                   key=lambda r: ic_connections.get(r, 0),
                                   reverse=True)
            # Add any unconnected components
            all_refs = set(mapper.get_all_refs()) - {ic_ref}
            for ref in all_refs - set(refs_to_place):
                refs_to_place.append(ref)
        else:
            refs_to_place = [r for r in mapper.get_all_refs() if r != ic_ref]

        # Place components in concentric rings based on connection strength
        ring_spacing = 2.5
        components_per_ring = 6

        for i, ref in enumerate(refs_to_place):
            fp = mapper.get_footprint(ref)
            if not fp:
                continue

            # Calculate ring and position
            ring = i // components_per_ring
            pos_in_ring = i % components_per_ring
            angle = (pos_in_ring * 2 * math.pi) / components_per_ring - math.pi / 2

            # Closer rings for more connected components
            radius = ring_spacing * (ring + 1)

            # Adjust angle based on which IC pad it connects to
            if analyzer:
                # Find which pads on this component connect to IC
                conn = analyzer.component_connectivity.get(ref)
                if conn:
                    ic_pads = [p for p in conn.pads
                              if any(p2.ref == ic_ref for p2 in
                                    analyzer.nets.get(p.net_name, []))]
                    # Could adjust angle based on IC pad positions here

            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)

            final_x, final_y = self._set_position_safe(fp, ref, x, y)
            positions[ref] = (final_x, final_y)
            placed += 1

        # Log wire length
        if analyzer:
            wire_length = analyzer.calculate_wire_length(positions)
            logger.info(f"  Estimated wire length: {wire_length:.1f}mm")

        return placed

    def _place_bridge_netlist(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                              center: Tuple[float, float], use_netlist: bool = True) -> int:
        """Place 2 components side by side based on pin connectivity"""
        placed = 0
        analyzer = mapper.netlist_analyzer if use_netlist else None
        positions: Dict[str, Tuple[float, float]] = {}

        refs = mapper.get_all_refs()
        if len(refs) != 2:
            return self._place_netlist_aware(mapper, rules, center)

        # For 2-component bridges, place them horizontally aligned
        # with shared nets on adjacent sides
        ref1, ref2 = refs[0], refs[1]
        fp1, fp2 = mapper.get_footprint(ref1), mapper.get_footprint(ref2)

        if fp1 and fp2:
            spacing = 3.5

            # Place first component
            x1, y1 = self._set_position_safe(fp1, ref1, center[0] - spacing/2, center[1])
            positions[ref1] = (x1, y1)
            placed += 1

            # Place second component
            x2, y2 = self._set_position_safe(fp2, ref2, center[0] + spacing/2, center[1])
            positions[ref2] = (x2, y2)
            placed += 1

            if analyzer:
                wire_length = analyzer.calculate_wire_length(positions)
                logger.info(f"  Estimated wire length: {wire_length:.1f}mm")

        return placed

    def _place_ic_centric(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                          center: Tuple[float, float]) -> int:
        """Place components radially around IC"""
        placed = 0

        # Find and place IC at center
        ic_comp = None
        for name, comp in rules.components.items():
            if comp.comp_type == ComponentType.IC:
                ic_comp = comp
                ref = mapper.get_ref_for_ato_name(name)
                if ref:
                    fp = mapper.get_footprint(ref)
                    if fp:
                        self._set_position_safe(fp, ref, center[0], center[1])
                        placed += 1
                        logger.debug(f"  Placed IC {ref} at center")
                break

        if not ic_comp:
            # No IC found, use generic placement
            return self._place_generic(mapper, rules, center)

        # Sort remaining components by priority
        # HF decoupling first (closest), then bulk, then others
        priority_order = [
            ComponentType.DECOUPLING_HF,
            ComponentType.DECOUPLING_BULK,
            ComponentType.CAPACITOR,
            ComponentType.RESISTOR,
            ComponentType.INDUCTOR,
            ComponentType.PASSIVE,
        ]

        sorted_comps = sorted(
            [(name, comp) for name, comp in rules.components.items()
             if comp.comp_type != ComponentType.IC],
            key=lambda x: (priority_order.index(x[1].comp_type)
                          if x[1].comp_type in priority_order else 99)
        )

        # Place components radially around IC
        angle_step = 2 * math.pi / max(len(sorted_comps), 1)
        current_angle = -math.pi / 2  # Start from bottom

        for name, comp in sorted_comps:
            ref = mapper.get_ref_for_ato_name(name)
            if not ref:
                # Try to find by matching pattern
                ref = self._find_ref_by_pattern(name, mapper)

            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    # Calculate position based on distance and angle
                    x = center[0] + comp.placement_distance * math.cos(current_angle)
                    y = center[1] + comp.placement_distance * math.sin(current_angle)

                    # Use safe placement with collision detection
                    final_x, final_y = self._set_position_safe(fp, ref, x, y)
                    placed += 1
                    logger.debug(f"  Placed {ref} at ({final_x:.1f}, {final_y:.1f})")

                    current_angle += angle_step

        return placed

    def _place_rf_linear(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                         center: Tuple[float, float]) -> int:
        """Place RF components in a linear chain"""
        placed = 0

        # Get all components and sort by type
        # Antenna goes at one end, then matching network components
        comps_by_type = {
            'antenna': [],
            'inductor': [],
            'capacitor': [],
            'other': []
        }

        for name, comp in rules.components.items():
            if comp.comp_type == ComponentType.CONNECTOR:
                comps_by_type['antenna'].append((name, comp))
            elif comp.comp_type in [ComponentType.INDUCTOR, ComponentType.RF_COMPONENT]:
                if 'l_' in name.lower():
                    comps_by_type['inductor'].append((name, comp))
                else:
                    comps_by_type['capacitor'].append((name, comp))
            elif comp.comp_type == ComponentType.CAPACITOR:
                comps_by_type['capacitor'].append((name, comp))
            else:
                comps_by_type['other'].append((name, comp))

        # Linear layout: RF_IN -> Inductors (series) -> Antenna
        # Capacitors go below as shunts
        x_pos = center[0] - 10  # Start left of center
        y_pos_main = center[1]
        y_pos_shunt = center[1] + 4  # Shunt caps below main path

        spacing = 4.0

        # Place series inductors
        for name, comp in comps_by_type['inductor']:
            ref = mapper.get_ref_for_ato_name(name) or self._find_ref_by_pattern(name, mapper)
            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    self._set_position_safe(fp, ref, x_pos, y_pos_main)
                    placed += 1
                    x_pos += spacing

        # Place antenna at end
        for name, comp in comps_by_type['antenna']:
            ref = mapper.get_ref_for_ato_name(name) or self._find_ref_by_pattern(name, mapper)
            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    self._set_position_safe(fp, ref, x_pos + 2, y_pos_main)
                    placed += 1

        # Place shunt capacitors below
        x_pos = center[0] - 6
        for name, comp in comps_by_type['capacitor']:
            ref = mapper.get_ref_for_ato_name(name) or self._find_ref_by_pattern(name, mapper)
            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    self._set_position_safe(fp, ref, x_pos, y_pos_shunt)
                    placed += 1
                    x_pos += 3.0

        return placed

    def _place_power_chain(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                           center: Tuple[float, float]) -> int:
        """Place power distribution components in a tree structure"""
        placed = 0

        # Group components
        bulk_caps = []
        ferrites = []
        decoupling = []
        other = []

        for name, comp in rules.components.items():
            name_lower = name.lower()
            if 'bulk' in name_lower:
                bulk_caps.append((name, comp))
            elif 'l_' in name_lower or comp.comp_type == ComponentType.INDUCTOR:
                ferrites.append((name, comp))
            elif comp.comp_type in [ComponentType.DECOUPLING_HF, ComponentType.DECOUPLING_BULK]:
                decoupling.append((name, comp))
            elif comp.comp_type == ComponentType.CAPACITOR:
                decoupling.append((name, comp))
            else:
                other.append((name, comp))

        # Layout:
        # BULK_CAP -> FERRITE_MAIN -> FERRITE_MCU  -> DECOUPLING_MCU
        #                          -> FERRITE_SENS -> DECOUPLING_SENS

        x_start = center[0] - 12
        y_center = center[1]

        # Place bulk cap at input
        x_pos = x_start
        for name, comp in bulk_caps:
            ref = mapper.get_ref_for_ato_name(name) or self._find_ref_by_pattern(name, mapper)
            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    self._set_position_safe(fp, ref, x_pos, y_center)
                    placed += 1
                    x_pos += 4

        # Place ferrites in sequence
        for name, comp in ferrites:
            ref = mapper.get_ref_for_ato_name(name) or self._find_ref_by_pattern(name, mapper)
            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    self._set_position_safe(fp, ref, x_pos, y_center)
                    placed += 1
                    x_pos += 3.5

        # Place decoupling caps in rows
        x_pos = center[0]
        y_offset = 3.5
        row = 0
        for name, comp in decoupling:
            ref = mapper.get_ref_for_ato_name(name) or self._find_ref_by_pattern(name, mapper)
            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    y = y_center + y_offset if row % 2 == 0 else y_center - y_offset
                    self._set_position_safe(fp, ref, x_pos, y)
                    placed += 1
                    if row % 2 == 1:
                        x_pos += 3.5
                    row += 1

        return placed

    def _place_bridge(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                      center: Tuple[float, float]) -> int:
        """Place 2 components side by side"""
        placed = 0
        x_pos = center[0] - 2

        for name, comp in rules.components.items():
            ref = mapper.get_ref_for_ato_name(name) or self._find_ref_by_pattern(name, mapper)
            if ref:
                fp = mapper.get_footprint(ref)
                if fp:
                    self._set_position_safe(fp, ref, x_pos, center[1])
                    placed += 1
                    x_pos += 4

        return placed

    def _place_generic(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                       center: Tuple[float, float]) -> int:
        """Generic grid-based placement"""
        placed = 0
        refs = mapper.get_all_refs()

        # Sort: ICs first, then by ref
        refs.sort(key=lambda r: (0 if r.startswith('U') else 1, r))

        grid_spacing = 3.5
        cols = max(int(math.sqrt(len(refs))), 2)

        for i, ref in enumerate(refs):
            fp = mapper.get_footprint(ref)
            if fp:
                col = i % cols
                row = i // cols
                x = center[0] + (col - cols/2) * grid_spacing
                y = center[1] + row * grid_spacing
                self._set_position_safe(fp, ref, x, y)
                placed += 1

        return placed

    def _place_netlist_aware(self, mapper: PCBComponentMapper, rules: ModuleDesignRules,
                             center: Tuple[float, float]) -> int:
        """Place components based on netlist connectivity to minimize wire length

        Uses force-directed placement:
        1. Get optimal placement order from netlist analyzer
        2. Place each component near its most connected already-placed neighbor
        3. Iteratively refine positions to minimize total wire length
        """
        if not mapper.netlist_analyzer:
            logger.debug("  No netlist analyzer available, using generic placement")
            return self._place_generic(mapper, rules, center)

        analyzer = mapper.netlist_analyzer
        placed = 0
        positions: Dict[str, Tuple[float, float]] = {}

        # Get optimal placement order
        placement_order = analyzer.get_optimal_placement_order()
        if not placement_order:
            return self._place_generic(mapper, rules, center)

        logger.debug(f"  Netlist-aware placement order: {placement_order[:5]}...")

        # Place first component at center
        first_ref = placement_order[0]
        fp = mapper.get_footprint(first_ref)
        if fp:
            x, y = self._set_position_safe(fp, first_ref, center[0], center[1])
            positions[first_ref] = (x, y)
            placed += 1
            logger.debug(f"  Placed {first_ref} at center ({x:.1f}, {y:.1f})")

        # Place remaining components near their most connected neighbors
        spacing = 3.5
        for ref in placement_order[1:]:
            fp = mapper.get_footprint(ref)
            if not fp:
                continue

            # Find best position based on connected components
            connections = analyzer.get_connected_components(ref)
            if connections:
                # Calculate weighted average position of connected components
                total_weight = 0
                weighted_x = 0.0
                weighted_y = 0.0

                for connected_ref, weight in connections.items():
                    if connected_ref in positions:
                        cx, cy = positions[connected_ref]
                        weighted_x += cx * weight
                        weighted_y += cy * weight
                        total_weight += weight

                if total_weight > 0:
                    target_x = weighted_x / total_weight
                    target_y = weighted_y / total_weight

                    # Add small offset based on placement order to avoid overlap
                    order_idx = placement_order.index(ref)
                    angle = (order_idx * 2 * math.pi) / len(placement_order)
                    offset_radius = spacing * (1 + order_idx * 0.1)
                    target_x += offset_radius * math.cos(angle)
                    target_y += offset_radius * math.sin(angle)
                else:
                    # No placed connections yet, use spiral from center
                    order_idx = placement_order.index(ref)
                    angle = (order_idx * 2 * math.pi) / len(placement_order)
                    radius = spacing * (1 + order_idx // 4)
                    target_x = center[0] + radius * math.cos(angle)
                    target_y = center[1] + radius * math.sin(angle)
            else:
                # No connections, place in spiral
                order_idx = placement_order.index(ref)
                angle = (order_idx * 2 * math.pi) / len(placement_order)
                radius = spacing * (1 + order_idx // 4)
                target_x = center[0] + radius * math.cos(angle)
                target_y = center[1] + radius * math.sin(angle)

            # Place with collision avoidance
            final_x, final_y = self._set_position_safe(fp, ref, target_x, target_y)
            positions[ref] = (final_x, final_y)
            placed += 1
            logger.debug(f"  Placed {ref} at ({final_x:.1f}, {final_y:.1f})")

        # Calculate and log total wire length
        wire_length = analyzer.calculate_wire_length(positions)
        logger.info(f"  Estimated wire length: {wire_length:.1f}mm")

        return placed

    def _get_current_positions(self, mapper: PCBComponentMapper) -> Dict[str, Tuple[float, float]]:
        """Get current positions of all components from the board

        Returns:
            Dict mapping ref -> (x, y) position in mm
        """
        positions = {}
        for ref in mapper.get_all_refs():
            fp = mapper.get_footprint(ref)
            if fp:
                pos = fp.GetPosition()
                positions[ref] = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        return positions

    def _refine_placement(self, mapper: PCBComponentMapper, center: Tuple[float, float],
                          iterations: int = 50, convergence_threshold: float = 0.1) -> bool:
        """Run force-directed optimization to refine component placement

        This method collects current positions, runs iterative force-directed
        optimization, and applies the refined positions.

        Args:
            mapper: PCB component mapper with netlist analyzer
            center: Center point for bounds checking
            iterations: Maximum number of optimization iterations
            convergence_threshold: Stop if max movement is below this (mm)

        Returns:
            True if refinement was applied, False if skipped
        """
        if not mapper.netlist_analyzer:
            logger.debug("  No netlist analyzer, skipping refinement")
            return False

        # Get current positions
        positions = self._get_current_positions(mapper)
        if len(positions) < 2:
            logger.debug("  Too few components for refinement")
            return False

        # Calculate initial wire length
        initial_wire_length = mapper.netlist_analyzer.calculate_wire_length(positions)
        logger.info(f"  Force-directed refinement: {iterations} iterations max")
        logger.debug(f"    Initial wire length: {initial_wire_length:.1f}mm")

        # Run optimization iterations
        for i in range(iterations):
            new_positions = self._optimize_placement_iteration(mapper, positions, center)

            # Check convergence (max movement)
            max_movement = 0.0
            for ref in positions:
                if ref in new_positions:
                    old_x, old_y = positions[ref]
                    new_x, new_y = new_positions[ref]
                    movement = math.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
                    max_movement = max(max_movement, movement)

            positions = new_positions

            if max_movement < convergence_threshold:
                logger.debug(f"    Converged after {i+1} iterations (movement={max_movement:.3f}mm)")
                break

        # Calculate final wire length
        final_wire_length = mapper.netlist_analyzer.calculate_wire_length(positions)
        improvement = initial_wire_length - final_wire_length
        logger.info(f"    Wire length: {initial_wire_length:.1f}mm -> {final_wire_length:.1f}mm "
                    f"({improvement:+.1f}mm)")

        # Only apply if there's improvement
        if final_wire_length >= initial_wire_length:
            logger.info("    No improvement from refinement, keeping original positions")
            return False

        # Apply refined positions with collision avoidance
        # Clear placement manager to re-register positions
        if self.placement_mgr:
            self.placement_mgr.placed_boxes.clear()

        for ref, (x, y) in positions.items():
            fp = mapper.get_footprint(ref)
            if fp:
                self._set_position_safe(fp, ref, x, y)

        return True

    def _optimize_placement_iteration(self, mapper: PCBComponentMapper,
                                       positions: Dict[str, Tuple[float, float]],
                                       center: Tuple[float, float]) -> Dict[str, Tuple[float, float]]:
        """Perform one iteration of placement optimization

        Uses force-directed approach: components are attracted to their
        connected components and repelled from overlapping.
        """
        if not mapper.netlist_analyzer:
            return positions

        analyzer = mapper.netlist_analyzer
        new_positions = positions.copy()
        
        # Pre-fetch component info to avoid repeated calls
        comp_info = {}
        for ref in positions:
            fp = mapper.get_footprint(ref)
            if fp:
                # Calculate approximate radius from bounding box
                bbox = self.placement_mgr.get_footprint_bbox(fp)
                width = bbox.x_max - bbox.x_min
                height = bbox.y_max - bbox.y_min
                radius = max(width, height) / 2.0
                
                # Check directly if locked
                is_locked = fp.IsLocked()
                
                comp_info[ref] = {
                    'radius': radius,
                    'locked': is_locked
                }
            else:
                comp_info[ref] = {'radius': 1.0, 'locked': False}

        # Constants for force calculation
        attraction_strength = 0.2  # Increased from 0.1
        repulsion_strength = 5.0   # Stronger repulsion
        damping = 0.4

        for ref, (x, y) in positions.items():
            # Skip if component is locked
            if comp_info[ref]['locked']:
                continue
                
            fx, fy = 0.0, 0.0
            my_radius = comp_info[ref]['radius']

            # Attraction forces from connected components
            connections = analyzer.get_connected_components(ref)
            for connected_ref, weight in connections.items():
                if connected_ref in positions:
                    cx, cy = positions[connected_ref]
                    dx = cx - x
                    dy = cy - y
                    dist = math.sqrt(dx*dx + dy*dy) + 0.1

                    # Attraction proportional to weight
                    # F = k * x (Hooke's law)
                    force = weight * attraction_strength
                    fx += force * dx / dist
                    fy += force * dy / dist

            # Repulsion forces from all nearby components (prevent overlap)
            for other_ref, (ox, oy) in positions.items():
                if other_ref == ref:
                    continue

                dx = x - ox
                dy = y - oy
                dist = math.sqrt(dx*dx + dy*dy) + 0.01 # Avoid div/0
                
                other_radius = comp_info[other_ref]['radius']
                min_separation = my_radius + other_radius + 0.5 # 0.5mm extra clearance
                
                if dist < min_separation:
                    # Repulsion force should be very strong when overlapping
                    # F = k / r^2
                    overlap_ratio = min_separation / dist
                    repulsion = repulsion_strength * (overlap_ratio ** 2)
                    fx += repulsion * dx / dist
                    fy += repulsion * dy / dist

            # Apply force with damping
            new_x = x + fx * damping
            new_y = y + fy * damping

            # Keep within reasonable bounds
            new_x = max(center[0] - 25, min(center[0] + 25, new_x))
            new_y = max(center[1] - 25, min(center[1] + 25, new_y))

            new_positions[ref] = (new_x, new_y)

        return new_positions

    def _refine_with_annealing(self, mapper: PCBComponentMapper, center: Tuple[float, float],
                                steps: int = 10000, t_max: float = 25.0, t_min: float = 0.1) -> bool:
        """Run simulated annealing optimization to refine component placement

        Simulated annealing can escape local minima better than force-directed
        methods, potentially finding better global solutions.

        Args:
            mapper: PCB component mapper with netlist analyzer
            center: Center point for bounds checking
            steps: Number of annealing steps
            t_max: Maximum (starting) temperature
            t_min: Minimum (ending) temperature

        Returns:
            True if refinement was applied, False if skipped
        """
        if not SIMANNEAL_AVAILABLE:
            logger.warning("  simanneal not installed, skipping annealing refinement")
            logger.warning("    Install with: pip install simanneal")
            return False

        if not mapper.netlist_analyzer:
            logger.debug("  No netlist analyzer, skipping annealing refinement")
            return False

        # Get current positions
        positions = self._get_current_positions(mapper)
        if len(positions) < 2:
            logger.debug("  Too few components for annealing refinement")
            return False

        # Calculate initial wire length
        initial_wire_length = mapper.netlist_analyzer.calculate_wire_length(positions)
        logger.info(f"  Simulated annealing refinement: {steps} steps")
        logger.debug(f"    Initial wire length: {initial_wire_length:.1f}mm")

        # Create and run the annealer
        annealer = PCBPlacementAnnealer(
            positions=positions,
            netlist_analyzer=mapper.netlist_analyzer,
            center=center,
            min_distance=2.0
        )
        annealer.Tmax = t_max
        annealer.Tmin = t_min
        annealer.steps = steps
        annealer.updates = 0  # Disable progress output

        # Run annealing
        final_positions, final_energy = annealer.anneal()

        # Calculate final wire length
        final_wire_length = mapper.netlist_analyzer.calculate_wire_length(final_positions)
        improvement = initial_wire_length - final_wire_length
        logger.info(f"    Wire length: {initial_wire_length:.1f}mm -> {final_wire_length:.1f}mm "
                    f"({improvement:+.1f}mm)")

        # Only apply if there's improvement
        if final_wire_length >= initial_wire_length:
            logger.info("    No improvement from annealing, keeping original positions")
            return False

        # Apply refined positions with collision avoidance
        if self.placement_mgr:
            self.placement_mgr.placed_boxes.clear()

        for ref, (x, y) in final_positions.items():
            fp = mapper.get_footprint(ref)
            if fp:
                self._set_position_safe(fp, ref, x, y)

        return True

    def _adjust_silkscreen_text(self, mapper: PCBComponentMapper) -> int:
        """Adjust silkscreen reference text to avoid overlapping components

        Checks each footprint's reference designator text and moves it if
        it overlaps with any component pads.

        Args:
            mapper: PCB component mapper

        Returns:
            Number of text elements adjusted
        """
        if not mapper.board:
            return 0

        adjusted = 0
        footprints = list(mapper.board.GetFootprints())

        # Build list of all pad bounding boxes for overlap checking
        all_pad_bounds = []
        for fp in footprints:
            for pad in fp.Pads():
                pos = pad.GetPosition()
                size = pad.GetSize()
                # Store as (min_x, min_y, max_x, max_y) in internal units
                bounds = (
                    pos.x - size.x // 2,
                    pos.y - size.y // 2,
                    pos.x + size.x // 2,
                    pos.y + size.y // 2
                )
                all_pad_bounds.append(bounds)

        # Also add component body bounds (approximate from footprint position and size)
        component_bounds = []
        for fp in footprints:
            bbox = fp.GetBoundingBox()
            # Use a smaller estimate for the body (not full courtyard)
            center_x = (bbox.GetLeft() + bbox.GetRight()) // 2
            center_y = (bbox.GetTop() + bbox.GetBottom()) // 2
            half_w = (bbox.GetRight() - bbox.GetLeft()) // 3  # Approximate body as 2/3 of bbox
            half_h = (bbox.GetBottom() - bbox.GetTop()) // 3
            component_bounds.append((
                center_x - half_w,
                center_y - half_h,
                center_x + half_w,
                center_y + half_h
            ))

        all_bounds = all_pad_bounds + component_bounds

        # Check and adjust each footprint's reference text
        for fp in footprints:
            ref = fp.GetReference()
            ref_text = fp.Reference()

            if not ref_text:
                continue

            # Get text bounding box
            text_bbox = ref_text.GetBoundingBox()
            text_bounds = (
                text_bbox.GetLeft(),
                text_bbox.GetTop(),
                text_bbox.GetRight(),
                text_bbox.GetBottom()
            )

            # Check for overlaps
            has_overlap = False
            for bounds in all_bounds:
                if self._bounds_overlap(text_bounds, bounds):
                    has_overlap = True
                    break

            if has_overlap:
                # Try to find a clear position
                fp_pos = fp.GetPosition()
                fp_bbox = fp.GetBoundingBox()

                # Calculate offsets to try (around the component)
                offsets = [
                    (0, -pcbnew.FromMM(2.5)),   # Above
                    (0, pcbnew.FromMM(2.5)),    # Below
                    (-pcbnew.FromMM(3.0), 0),   # Left
                    (pcbnew.FromMM(3.0), 0),    # Right
                    (-pcbnew.FromMM(2.0), -pcbnew.FromMM(2.0)),  # Top-left
                    (pcbnew.FromMM(2.0), -pcbnew.FromMM(2.0)),   # Top-right
                    (-pcbnew.FromMM(2.0), pcbnew.FromMM(2.0)),   # Bottom-left
                    (pcbnew.FromMM(2.0), pcbnew.FromMM(2.0)),    # Bottom-right
                ]

                best_pos = None
                for dx, dy in offsets:
                    test_x = fp_pos.x + dx
                    test_y = fp_pos.y + dy

                    # Create test bounds for text at this position
                    text_w = text_bbox.GetRight() - text_bbox.GetLeft()
                    text_h = text_bbox.GetBottom() - text_bbox.GetTop()
                    test_bounds = (
                        test_x - text_w // 2,
                        test_y - text_h // 2,
                        test_x + text_w // 2,
                        test_y + text_h // 2
                    )

                    # Check if this position is clear
                    is_clear = True
                    for bounds in all_bounds:
                        if self._bounds_overlap(test_bounds, bounds):
                            is_clear = False
                            break

                    if is_clear:
                        best_pos = (test_x, test_y)
                        break

                if best_pos:
                    # Move text to clear position
                    new_pos = pcbnew.VECTOR2I(int(best_pos[0]), int(best_pos[1]))
                    ref_text.SetPosition(new_pos)
                    adjusted += 1
                    logger.debug(f"    Moved {ref} silkscreen to clear position")

        if adjusted > 0:
            logger.info(f"  Adjusted {adjusted} silkscreen text positions")

        return adjusted

    def _bounds_overlap(self, bounds1: Tuple, bounds2: Tuple) -> bool:
        """Check if two bounds tuples overlap

        Args:
            bounds1, bounds2: Tuples of (min_x, min_y, max_x, max_y)

        Returns:
            True if they overlap
        """
        margin = pcbnew.FromMM(0.2)  # Small clearance margin
        return not (
            bounds1[2] + margin < bounds2[0] or  # bounds1 right < bounds2 left
            bounds1[0] - margin > bounds2[2] or  # bounds1 left > bounds2 right
            bounds1[3] + margin < bounds2[1] or  # bounds1 bottom < bounds2 top
            bounds1[1] - margin > bounds2[3]     # bounds1 top > bounds2 bottom
        )

    def _fallback_placement(self, module_name: str, center: Tuple[float, float]) -> int:
        """Fallback to simple grid placement if parsing fails"""
        pcb_path = os.path.join(
            self.project_dir, "layouts", module_name, f"{module_name}.kicad_pcb"
        )
        mapper = PCBComponentMapper(pcb_path)

        if not mapper.board:
            return 0

        return self._place_generic(mapper,
                                   ModuleDesignRules(module_name, ModuleTopology.GENERIC, {}, []),
                                   center)

    def _find_ref_by_pattern(self, ato_name: str, mapper: PCBComponentMapper) -> Optional[str]:
        """Try to find reference by matching patterns in ato name"""
        # Map common atopile naming to KiCad prefixes
        prefix_map = {
            'c_': 'C',
            'cp': 'C',
            'r_': 'R',
            'l_': 'L',
            'ic': 'U',
            'ant': 'ANT',
            'led': 'LED',
        }

        ato_lower = ato_name.lower()
        for pattern, prefix in prefix_map.items():
            if ato_lower.startswith(pattern):
                # Find matching refs
                for ref in mapper.get_all_refs():
                    if ref.startswith(prefix):
                        return ref

        return None

    def _set_position(self, fp, x_mm: float, y_mm: float):
        """Set footprint position in mm"""
        pos = pcbnew.VECTOR2I(
            int(pcbnew.FromMM(x_mm)),
            int(pcbnew.FromMM(y_mm))
        )
        fp.SetPosition(pos)

    def _set_position_safe(self, fp, ref: str, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """Set footprint position with collision detection

        Returns:
            Tuple of final (x, y) position in mm
        """
        if self.placement_mgr:
            # Find non-overlapping position
            final_x, final_y = self.placement_mgr.find_non_overlapping_position(
                fp, ref, x_mm, y_mm
            )
            # Set position
            self._set_position(fp, final_x, final_y)
            # Register placement for future collision checks
            self.placement_mgr.register_placement(ref, fp)
            return final_x, final_y
        else:
            # No placement manager, just set position
            self._set_position(fp, x_mm, y_mm)
            return x_mm, y_mm


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Smart Component Placer using Atopile Design Rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Place all modules
    python smart_placer.py

    # Place specific modules
    python smart_placer.py --modules power_filter,rf_matching

    # Verbose output showing parsed rules
    python smart_placer.py --verbose

Note: Run with KiCad's Python interpreter:
    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3 smart_placer.py
"""
    )
    parser.add_argument('--modules', default='all',
                        help='Comma-separated list of modules, or "all"')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze and print design rules, do not place')
    parser.add_argument('--refine', choices=['force', 'anneal', 'none'], default='anneal',
                        help='Refinement method: anneal (simulated annealing, default), force (force-directed), none')

    args = parser.parse_args()

    # Check simanneal availability if requested
    if args.refine == 'anneal' and not SIMANNEAL_AVAILABLE:
        logger.warning("simanneal not installed, falling back to force-directed")
        logger.warning("Install with: pip install simanneal")
        args.refine = 'force'

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not PCBNEW_AVAILABLE and not args.analyze_only:
        logger.error("pcbnew required for placement")
        logger.error("Run with KiCad's Python interpreter")
        sys.exit(1)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    placer = SmartPlacer(project_dir)

    # Get module list
    layouts_dir = os.path.join(project_dir, "layouts")
    # Skip non-module layouts
    skip_layouts = {'default', 'example'}
    all_modules = [d for d in os.listdir(layouts_dir)
                   if os.path.isdir(os.path.join(layouts_dir, d))
                   and d not in skip_layouts]

    if args.modules == 'all':
        modules = all_modules
    else:
        modules = [m.strip() for m in args.modules.split(',')]

    print(f"\n{'='*60}")
    print("Smart Component Placer")
    print(f"{'='*60}")

    results = {}
    for module_name in sorted(modules):
        print(f"\n--- {module_name} ---")

        # Parse and optionally show design rules
        design_rules = placer.parser.parse_module(module_name)
        if design_rules:
            print(f"  Topology: {design_rules.topology.value}")
            print(f"  Components: {len(design_rules.components)}")
            if args.verbose:
                for name, comp in design_rules.components.items():
                    print(f"    {name}: {comp.comp_type.value} "
                          f"(dist={comp.placement_distance:.1f}mm)")

        if not args.analyze_only:
            placed = placer.place_module(module_name, refine_method=args.refine)
            results[module_name] = placed
            print(f"  Placed: {placed} components")
        else:
            print(f"  (analyze only - no placement)")

    if not args.analyze_only:
        print(f"\n{'='*60}")
        print("Placement Summary")
        print(f"{'='*60}")
        for name, count in sorted(results.items()):
            status = "✓" if count > 0 else "✗"
            print(f"  {status} {name:20} {count} components")


if __name__ == "__main__":
    main()
