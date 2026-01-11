#!/usr/bin/env python3
"""
Advanced KiCad PCB Component Auto-Placer v2.0

IMPROVEMENTS FROM REVIEW:
- Uses KiCad pcbnew Python API for robust PCB file handling (no regex)
- Quadtree spatial partitioning for O(n log n) collision detection
- YAML schema validation with descriptive errors
- Rotation optimization (tests 0/90/180/270)
- Proper error handling and logging
- Reproducible random seeds
- Configurable magic numbers

Features:
- YAML configuration for easy editing
- Component grouping with relative positioning
- Force-directed algorithm with netlist-aware attraction
- Bounding box collision detection with layer awareness
- Post-optimization overlap resolution

Usage:
    python autoplace_v2.py [--config placement_config.yaml] [--optimize]
    
Requirements:
    - KiCad 8.0+ (for pcbnew Python bindings)
    - pyyaml
"""

import os
import sys
import math
import random
import argparse
import logging
import glob as glob_module
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    logger.info("Installing pyyaml...")
    os.system(f"{sys.executable} -m pip install pyyaml")
    import yaml

# Try to import pcbnew (KiCad Python bindings)
PCBNEW_AVAILABLE = False
try:
    import pcbnew
    PCBNEW_AVAILABLE = True
    logger.info("Using KiCad pcbnew API for robust PCB handling")
except ImportError:
    logger.warning("pcbnew not available - falling back to regex parsing")
    logger.warning("For best results, run from KiCad's Python environment")

# Try to import matplotlib for visualization
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass  # MATPLOTLIB_AVAILABLE remains False
    logger.debug("matplotlib not available - visualization disabled")


# ============================================================================
# CONFIGURATION & VALIDATION
# ============================================================================

REQUIRED_YAML_FIELDS = {'board', 'groups'}
OPTIONAL_YAML_FIELDS = {'force_directed', 'connections', 'component_sizes'}

YAML_SCHEMA = """
YAML Configuration Schema:
  board:
    width: float (mm)
    height: float (mm)
    origin: [float, float] (optional)
  
  groups:
    <group_name>:
      anchor: str (reference designator)
      position: [float, float] (mm from origin)
      layer: str (front/back, optional)
      components:
        <ref>:
          offset: [float, float]
          rotation: float (optional)
          size: [float, float] (optional)
          layer: str (optional)
  
  force_directed:
    enabled: bool
    iterations: int
    attraction_strength: float
    repulsion_strength: float
    damping: float
    collision_strength: float
    min_clearance: float
    random_seed: int (optional, for reproducibility)
  
  connections: (optional)
    - [ref1, ref2, weight]
"""


class ConfigValidationError(Exception):
    """Raised when YAML configuration is invalid."""
    pass


def validate_config(config: dict) -> None:
    """Validate YAML configuration schema."""
    # Check required fields
    for field_name in REQUIRED_YAML_FIELDS:
        if field_name not in config:
            raise ConfigValidationError(f"Missing required field: '{field_name}'\n{YAML_SCHEMA}")
    
    # Validate board
    board = config.get('board', {})
    if 'width' not in board or 'height' not in board:
        raise ConfigValidationError("board must have 'width' and 'height'")
    
    # Validate groups
    groups = config.get('groups', {})
    if not groups:
        raise ConfigValidationError("'groups' cannot be empty")
    
    for group_name, group_data in groups.items():
        if 'position' not in group_data:
            raise ConfigValidationError(f"Group '{group_name}' missing 'position'")
        if 'components' not in group_data:
            raise ConfigValidationError(f"Group '{group_name}' missing 'components'")
    
    logger.info("Configuration validated successfully")


# ============================================================================
# QUADTREE FOR SPATIAL PARTITIONING
# ============================================================================

@dataclass
class Rectangle:
    """Axis-aligned bounding box."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def left(self) -> float:
        return self.x - self.width / 2
    
    @property
    def right(self) -> float:
        return self.x + self.width / 2
    
    @property
    def top(self) -> float:
        return self.y + self.height / 2
    
    @property
    def bottom(self) -> float:
        return self.y - self.height / 2
    
    def contains(self, x: float, y: float) -> bool:
        return (self.left <= x <= self.right and
                self.bottom <= y <= self.top)
    
    def intersects(self, other: 'Rectangle') -> bool:
        return not (self.right < other.left or
                    self.left > other.right or
                    self.top < other.bottom or
                    self.bottom > other.top)


class Quadtree:
    """
    Quadtree for O(n log n) collision detection.
    
    Instead of checking all n² pairs, we partition space and only
    check components in the same or adjacent cells.
    """
    
    MAX_OBJECTS = 4
    MAX_LEVELS = 8
    
    def __init__(self, bounds: Rectangle, level: int = 0):
        self.bounds = bounds
        self.level = level
        self.objects: List[Tuple[str, Rectangle]] = []  # (ref, bounds)
        self.nodes: List[Optional['Quadtree']] = [None, None, None, None]
    
    def clear(self):
        """Clear the quadtree."""
        self.objects.clear()
        for i, node in enumerate(self.nodes):
            if node:
                node.clear()
                self.nodes[i] = None
    
    def split(self):
        """Split into 4 quadrants."""
        sub_w = self.bounds.width / 2
        sub_h = self.bounds.height / 2
        x = self.bounds.x
        y = self.bounds.y
        
        # NE, NW, SW, SE
        self.nodes[0] = Quadtree(Rectangle(x + sub_w/2, y + sub_h/2, sub_w, sub_h), self.level + 1)
        self.nodes[1] = Quadtree(Rectangle(x - sub_w/2, y + sub_h/2, sub_w, sub_h), self.level + 1)
        self.nodes[2] = Quadtree(Rectangle(x - sub_w/2, y - sub_h/2, sub_w, sub_h), self.level + 1)
        self.nodes[3] = Quadtree(Rectangle(x + sub_w/2, y - sub_h/2, sub_w, sub_h), self.level + 1)
    
    def get_index(self, rect: Rectangle) -> int:
        """Get quadrant index for a rectangle (-1 if it spans multiple)."""
        mid_x = self.bounds.x
        mid_y = self.bounds.y
        
        top_half = rect.bottom > mid_y
        bottom_half = rect.top < mid_y
        left_half = rect.right < mid_x
        right_half = rect.left > mid_x
        
        if top_half:
            if right_half:
                return 0  # NE
            elif left_half:
                return 1  # NW
        elif bottom_half:
            if left_half:
                return 2  # SW
            elif right_half:
                return 3  # SE
        
        return -1  # Spans multiple quadrants
    
    def insert(self, ref: str, rect: Rectangle):
        """Insert a component into the quadtree."""
        if self.nodes[0] is not None:
            index = self.get_index(rect)
            if index != -1:
                self.nodes[index].insert(ref, rect)
                return
        
        self.objects.append((ref, rect))
        
        if len(self.objects) > self.MAX_OBJECTS and self.level < self.MAX_LEVELS:
            if self.nodes[0] is None:
                self.split()
            
            i = 0
            while i < len(self.objects):
                index = self.get_index(self.objects[i][1])
                if index != -1:
                    obj = self.objects.pop(i)
                    self.nodes[index].insert(obj[0], obj[1])
                else:
                    i += 1
    
    def retrieve(self, rect: Rectangle) -> List[Tuple[str, Rectangle]]:
        """Retrieve all objects that could collide with the given rectangle."""
        result = list(self.objects)
        
        if self.nodes[0] is not None:
            index = self.get_index(rect)
            if index != -1:
                result.extend(self.nodes[index].retrieve(rect))
            else:
                # Spans multiple - check all children
                for node in self.nodes:
                    if node and node.bounds.intersects(rect):
                        result.extend(node.retrieve(rect))
        
        return result


# ============================================================================
# COMPONENT MODEL
# ============================================================================

@dataclass
class Component:
    """Represents a component with position, size, and properties."""
    ref: str
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    width: float = 2.0
    height: float = 2.0
    layer: str = "front"
    group: Optional[str] = None
    note: str = ""
    vx: float = 0.0
    vy: float = 0.0
    fixed: bool = False
    
    def get_bounds(self) -> Rectangle:
        """Get bounding rectangle."""
        return Rectangle(self.x, self.y, self.width, self.height)
    
    @property
    def left(self) -> float:
        return self.x - self.width / 2
    
    @property
    def right(self) -> float:
        return self.x + self.width / 2
    
    @property
    def top(self) -> float:
        return self.y + self.height / 2
    
    @property
    def bottom(self) -> float:
        return self.y - self.height / 2
    
    def overlaps(self, other: 'Component', margin: float = 0.5) -> bool:
        """Check if this component overlaps with another."""
        if self.layer != other.layer:
            return False
        return (
            self.left - margin < other.right + margin and
            self.right + margin > other.left - margin and
            self.bottom - margin < other.top + margin and
            self.top + margin > other.bottom - margin
        )
    
    def get_base_dimensions(self) -> Tuple[float, float]:
        """Get the base (unrotated) dimensions of this component.

        Returns (base_width, base_height) as if rotation were 0.
        """
        if self.rotation in [90, 270]:
            # Dimensions are currently swapped, so swap back
            return (self.height, self.width)
        else:
            return (self.width, self.height)

    def get_optimal_rotation(self, neighbors: List['Component'],
                              rotations: Optional[List[float]] = None) -> float:
        """Find rotation that minimizes overlap with neighbors."""
        if rotations is None:
            rotations = [0, 90, 180, 270]

        best_rotation = self.rotation
        min_overlap = float('inf')

        # Get base dimensions (as if rotation=0)
        base_w, base_h = self.get_base_dimensions()
        original_w, original_h = self.width, self.height

        for rot in rotations:
            # Set dimensions based on target rotation
            if rot in [90, 270]:
                self.width, self.height = base_h, base_w
            else:
                self.width, self.height = base_w, base_h

            overlap_count = sum(1 for n in neighbors if self.overlaps(n, margin=0.1))

            if overlap_count < min_overlap:
                min_overlap = overlap_count
                best_rotation = rot

        # Restore original dimensions
        self.width, self.height = original_w, original_h
        return best_rotation


@dataclass
class ComponentGroup:
    """A group of related components with relative positioning."""
    name: str
    anchor: str
    position: Tuple[float, float]
    layer: str = "front"
    components: Dict[str, dict] = field(default_factory=dict)


@dataclass
class ForceDirectedConfig:
    """Configuration for force-directed placement."""
    enabled: bool = True
    iterations: int = 200
    attraction_strength: float = 0.15
    repulsion_strength: float = 500.0
    damping: float = 0.82
    collision_strength: float = 50.0
    min_clearance: float = 0.5
    random_seed: Optional[int] = 42  # For reproducibility
    max_velocity: float = 3.0
    auto_rotate: bool = True  # Enable rotation optimization
    rotation_interval: int = 50  # Run rotation optimization every N iterations
    board_edge_margin: float = 1.0  # mm margin from board edges
    overlap_push_factor: float = 0.5  # Additional push when resolving overlaps


# ============================================================================
# COMPONENT SIZE DATABASE
# ============================================================================

COMPONENT_SIZES = {
    # ICs and Modules
    "U1": (14.0, 14.0),    # RAK3172-SIP
    "U2": (4.5, 4.5),      # LIS3DHTR LGA-16
    "U3": (4.0, 2.5),      # FT24C02A SOT-23-5
    "U4": (4.5, 4.0),      # QMI8658C LGA-14
    "U5": (4.5, 4.5),      # HDC2080 WSON-6
    
    # Battery
    "B1": (26.0, 26.0),    # CR2032 holder
    
    # Antenna
    "ANT1": (10.0, 5.0),   # SMD antenna
    
    # Connectors
    "H1": (15.0, 4.0),     # 5-pin debug header
    
    # LED
    "LED1": (2.5, 1.5),    # 0603 LED
    
    # Default sizes
    "_resistor": (1.8, 1.2),    # 0402
    "_capacitor": (1.8, 1.2),   # 0402
    "_inductor": (2.5, 2.0),    # 0603
}


def get_component_size(ref: str) -> Tuple[float, float]:
    """Get component size from database or estimate from prefix."""
    if ref in COMPONENT_SIZES:
        return COMPONENT_SIZES[ref]
    
    prefix = ''.join(c for c in ref if c.isalpha())
    size_map = {
        'R': COMPONENT_SIZES['_resistor'],
        'C': COMPONENT_SIZES['_capacitor'],
        'L': COMPONENT_SIZES['_inductor'],
        'D': (2.0, 1.5),
    }
    return size_map.get(prefix, (2.0, 2.0))


# ============================================================================
# FORCE-DIRECTED PLACER (with Quadtree optimization)
# ============================================================================

class ForceDirectedPlacer:
    """Force-directed placement with O(n log n) collision detection."""
    
    def __init__(self, config: ForceDirectedConfig, board_width: float, 
                 board_height: float):
        self.config = config
        self.board_width = board_width
        self.board_height = board_height
        self.components: Dict[str, Component] = {}
        self.connections: List[Tuple[str, str, float]] = []
        self.quadtree: Optional[Quadtree] = None
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            logger.info(f"Random seed set to {config.random_seed}")
    
    def add_component(self, component: Component):
        self.components[component.ref] = component
    
    def add_connection(self, ref1: str, ref2: str, weight: float = 1.0):
        if ref1 in self.components and ref2 in self.components:
            self.connections.append((ref1, ref2, weight))
    
    def build_quadtree(self, layer: str = "front"):
        """Build quadtree for efficient collision detection."""
        bounds = Rectangle(0, 0, self.board_width * 2, self.board_height * 2)
        self.quadtree = Quadtree(bounds)
        
        for ref, comp in self.components.items():
            if comp.layer == layer:
                self.quadtree.insert(ref, comp.get_bounds())
    
    def get_nearby_components(self, comp: Component) -> List[Component]:
        """Get components that might collide (using quadtree)."""
        if self.quadtree is None:
            # Fallback to O(n) scan
            return [c for c in self.components.values() 
                    if c.ref != comp.ref and c.layer == comp.layer]
        
        rect = Rectangle(comp.x, comp.y, 
                         comp.width + self.config.min_clearance * 4,
                         comp.height + self.config.min_clearance * 4)
        candidates = self.quadtree.retrieve(rect)
        return [self.components[ref] for ref, _ in candidates 
                if ref != comp.ref and ref in self.components]
    
    def calculate_forces(self, layer: str = "front") -> Dict[str, Tuple[float, float]]:
        """Calculate all forces on each component.

        Args:
            layer: The layer to calculate forces for. Quadtree should be built
                   for this layer before calling.
        """
        forces = {ref: [0.0, 0.0] for ref in self.components}

        # Attraction forces (from connections) - these work across layers
        for ref1, ref2, weight in self.connections:
            c1, c2 = self.components[ref1], self.components[ref2]
            dx = c2.x - c1.x
            dy = c2.y - c1.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > 0.1:
                force = self.config.attraction_strength * weight * distance
                fx = (dx / distance) * force
                fy = (dy / distance) * force

                if not c1.fixed:
                    forces[ref1][0] += fx
                    forces[ref1][1] += fy
                if not c2.fixed:
                    forces[ref2][0] -= fx
                    forces[ref2][1] -= fy

        # Repulsion and collision forces (using quadtree for efficiency)
        # Only process components on the specified layer
        for ref, comp in self.components.items():
            if comp.fixed or comp.layer != layer:
                continue

            nearby = self.get_nearby_components(comp)

            for other in nearby:
                dx = comp.x - other.x
                dy = comp.y - other.y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < 0.1:
                    angle = random.uniform(0, 2 * math.pi)
                    dx, dy = math.cos(angle), math.sin(angle)
                    distance = 0.1

                # Collision force
                if comp.overlaps(other, margin=self.config.min_clearance):
                    overlap_x = min(comp.right, other.right) - max(comp.left, other.left)
                    overlap_y = min(comp.top, other.top) - max(comp.bottom, other.bottom)
                    overlap = max(overlap_x, overlap_y) + self.config.min_clearance

                    force = self.config.collision_strength * overlap
                    forces[ref][0] += (dx / distance) * force
                    forces[ref][1] += (dy / distance) * force
                else:
                    # Repulsion force
                    min_dist = (comp.width + other.width) / 2 + self.config.min_clearance
                    if distance < min_dist * 2:
                        force = self.config.repulsion_strength / (distance * distance)
                        forces[ref][0] += (dx / distance) * force
                        forces[ref][1] += (dy / distance) * force

        return {ref: tuple(f) for ref, f in forces.items()}
    
    def constrain_to_board(self, comp: Component):
        """Keep component within board boundaries."""
        edge_margin = self.config.board_edge_margin
        margin_x = comp.width / 2 + edge_margin
        margin_y = comp.height / 2 + edge_margin

        half_w = self.board_width / 2 - margin_x
        half_h = self.board_height / 2 - margin_y

        comp.x = max(-half_w, min(half_w, comp.x))
        comp.y = max(-half_h, min(half_h, comp.y))
    
    def count_overlaps(self) -> int:
        """Count overlapping component pairs using quadtree for O(n log n).
        
        Checks both front and back layers.
        """
        overlaps = 0
        checked_pairs = set()  # Avoid double-counting
        
        # Check overlaps per layer
        for layer in ["front", "back"]:
            # Build quadtree for this layer
            self.build_quadtree(layer)
            
            # Only check components on this layer
            layer_components = {ref: comp for ref, comp in self.components.items() 
                              if comp.layer == layer}
            
            for ref, comp in layer_components.items():
                # Get nearby candidates from quadtree
                nearby = self.get_nearby_components(comp)
                
                for other in nearby:
                    # Create sorted pair key to avoid duplicates
                    pair_key = tuple(sorted([ref, other.ref]))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)
                    
                    if comp.overlaps(other, margin=0):
                        overlaps += 1
        
        return overlaps
    
    def optimize_rotations(self):
        """Optimize component rotations to minimize overlaps."""
        rotations_changed = 0

        for ref, comp in self.components.items():
            if comp.fixed:
                continue

            # Get nearby components
            neighbors = self.get_nearby_components(comp)
            if not neighbors:
                continue

            # Get base dimensions (unrotated)
            base_w, base_h = comp.get_base_dimensions()

            # Find best rotation
            best_rot = comp.get_optimal_rotation(neighbors)

            if best_rot != comp.rotation:
                # Apply rotation with correct dimension handling
                if best_rot in [90, 270]:
                    comp.width = base_h
                    comp.height = base_w
                else:  # 0, 180
                    comp.width = base_w
                    comp.height = base_h

                comp.rotation = best_rot
                rotations_changed += 1

        return rotations_changed
    
    def run(self) -> Dict[str, Component]:
        """Run force-directed placement with auto-rotation."""
        if not self.config.enabled:
            logger.info("Force-directed optimization disabled")
            return self.components
        
        logger.info(f"Running force-directed placement ({self.config.iterations} iterations)")
        if self.config.auto_rotate:
            logger.info(f"  Auto-rotation enabled (every {self.config.rotation_interval} iterations)")
        
        initial_overlaps = self.count_overlaps()
        logger.info(f"  Initial overlaps: {initial_overlaps}")
        
        for iteration in range(self.config.iterations):
            # Calculate forces for each layer separately
            # Build quadtree and calculate forces per layer to handle both front and back
            all_forces = {ref: [0.0, 0.0] for ref in self.components}

            for layer in ["front", "back"]:
                self.build_quadtree(layer)
                layer_forces = self.calculate_forces(layer)
                for ref, (fx, fy) in layer_forces.items():
                    all_forces[ref][0] += fx
                    all_forces[ref][1] += fy

            forces = {ref: tuple(f) for ref, f in all_forces.items()}
            
            # Apply forces
            for ref, (fx, fy) in forces.items():
                comp = self.components[ref]
                if comp.fixed:
                    continue
                
                comp.vx = (comp.vx + fx) * self.config.damping
                comp.vy = (comp.vy + fy) * self.config.damping
                
                speed = math.sqrt(comp.vx**2 + comp.vy**2)
                if speed > self.config.max_velocity:
                    comp.vx = (comp.vx / speed) * self.config.max_velocity
                    comp.vy = (comp.vy / speed) * self.config.max_velocity
                
                comp.x += comp.vx
                comp.y += comp.vy
                
                self.constrain_to_board(comp)
            
            # Auto-rotation optimization
            if self.config.auto_rotate and iteration > 0 and iteration % self.config.rotation_interval == 0:
                rotations = self.optimize_rotations()
                if rotations > 0:
                    logger.info(f"  Iteration {iteration}: Rotated {rotations} components")
            
            if iteration % 40 == 0:
                overlaps = self.count_overlaps()
                logger.info(f"  Iteration {iteration}: Overlaps={overlaps}")
        
        # Final rotation pass
        if self.config.auto_rotate:
            final_rotations = self.optimize_rotations()
            logger.info(f"  Final rotation pass: {final_rotations} adjustments")
        
        final_overlaps = self.count_overlaps()
        logger.info(f"  Force-directed complete. Final overlaps: {final_overlaps}")
        
        if final_overlaps > 0:
            logger.warning(f"  {final_overlaps} overlaps remain - running resolution")
        
        return self.components


# ============================================================================
# NETLIST PARSER (Extract connections from pcbnew)
# ============================================================================

class NetlistParser:
    """Parses KiCad PCB netlists to extract component connections.
    
    Uses pcbnew API to read actual nets, avoiding manual YAML connection lists.
    Weights are assigned based on net type (power=higher, signals=lower).
    """
    
    # Net name patterns for classification
    POWER_PATTERNS = ['vcc', 'vdd', 'v3.3', '3v3', '5v', 'vbat', 'pwr']
    GROUND_PATTERNS = ['gnd', 'vss', 'ground', 'agnd', 'dgnd']
    HIGH_SPEED_PATTERNS = ['clk', 'clock', 'spi', 'i2c', 'uart', 'usb', 'rf']
    
    def __init__(self, board):
        """Initialize with a pcbnew board object."""
        self.board = board
        self.connections: List[Tuple[str, str, float]] = []
        self.component_refs: Set[str] = set()
    
    @classmethod
    def from_pcb_file(cls, pcb_path: str) -> Optional['NetlistParser']:
        """Create parser from PCB file path."""
        if not PCBNEW_AVAILABLE:
            logger.warning("pcbnew not available - cannot parse netlist")
            return None
        
        try:
            board = pcbnew.LoadBoard(pcb_path)
            return cls(board)
        except Exception as e:
            logger.error(f"Failed to load PCB for netlist: {e}")
            return None
    
    def classify_net(self, net_name: str) -> Tuple[str, float]:
        """Classify net type and return (type, weight).
        
        Returns:
            Tuple of (net_type, base_weight)
            - power: High weight (3.0) - keep decoupling close
            - ground: Medium weight (2.0) - many connections
            - high_speed: High weight (2.5) - minimize trace length
            - signal: Normal weight (1.0)
        """
        name_lower = net_name.lower()
        
        for pattern in self.POWER_PATTERNS:
            if pattern in name_lower:
                return ('power', 3.0)
        
        for pattern in self.GROUND_PATTERNS:
            if pattern in name_lower:
                return ('ground', 2.0)
        
        for pattern in self.HIGH_SPEED_PATTERNS:
            if pattern in name_lower:
                return ('high_speed', 2.5)
        
        return ('signal', 1.0)
    
    def _sample_power_net_connections(self, refs: List[str], weight: float,
                                       max_connections: int) -> List[Tuple[str, str, float]]:
        """Sample representative connections from a large power/ground net.

        Strategy: Prioritize connections between different component types,
        especially decoupling capacitors (C*) near ICs (U*).

        Args:
            refs: List of component references on this net
            weight: Base weight for connections
            max_connections: Maximum number of connections to return

        Returns:
            List of (ref1, ref2, weight) tuples
        """
        connections = []

        # Categorize components
        ics = [r for r in refs if r.startswith('U')]
        caps = [r for r in refs if r.startswith('C')]
        # Note: 'others' could be used for additional connection strategies
        
        # Priority 1: Connect each cap to nearest IC (decoupling placement)
        # Use higher weight for these critical connections
        decoupling_weight = weight * 1.5
        for cap in caps:
            if ics and len(connections) < max_connections:
                # Connect to first IC (in real use, would pick nearest)
                ic = ics[0] if len(ics) == 1 else random.choice(ics)
                connections.append((cap, ic, decoupling_weight))
        
        # Priority 2: Connect ICs to each other (if multiple)
        if len(ics) > 1:
            for i in range(min(len(ics) - 1, max_connections - len(connections))):
                if i + 1 < len(ics):
                    connections.append((ics[i], ics[i + 1], weight))
        
        # Priority 3: Random sampling of remaining pairs
        remaining_budget = max_connections - len(connections)
        if remaining_budget > 0 and len(refs) > 1:
            # Create pool of unused pairs
            used_pairs = {tuple(sorted([c[0], c[1]])) for c in connections}
            potential_pairs = []
            
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    pair = tuple(sorted([refs[i], refs[j]]))
                    if pair not in used_pairs:
                        potential_pairs.append((refs[i], refs[j]))
            
            # Random sample from remaining
            if potential_pairs:
                sample_size = min(remaining_budget, len(potential_pairs))
                sampled = random.sample(potential_pairs, sample_size)
                for ref1, ref2 in sampled:
                    connections.append((ref1, ref2, weight * 0.5))  # Lower weight for random pairs
        
        return connections
    
    def parse(self, max_pins_per_net: int = 10, 
              power_net_sample_size: int = 20) -> List[Tuple[str, str, float]]:
        """Extract connections from netlist.
        
        Args:
            max_pins_per_net: For signal nets, skip if more pins than this.
            power_net_sample_size: For power/ground nets, sample this many 
                                   representative connections instead of skipping.
        
        Returns:
            List of (ref1, ref2, weight) tuples for force-directed placer.
        """
        if self.board is None:
            return []
        
        self.connections.clear()
        self.component_refs.clear()
        
        try:
            # Get all nets
            nets = self.board.GetNetsByName()
            sampled_power_nets = 0
            
            for net_name, net in nets.items():
                if not net_name or net_name == "":
                    continue
                
                # Get pads connected to this net
                pads = list(net.Pads())
                
                if len(pads) < 2:
                    continue
                
                # Classify net
                net_type, base_weight = self.classify_net(net_name)
                
                # Collect unique component refs
                refs_in_net = []
                for pad in pads:
                    fp = pad.GetParent()
                    if fp:
                        ref = fp.GetReference()
                        refs_in_net.append(ref)
                        self.component_refs.add(ref)
                
                # Remove duplicates (same component, multiple pads)
                unique_refs = list(set(refs_in_net))
                
                # Handle large nets differently based on type
                if len(pads) > max_pins_per_net:
                    if net_type in ['power', 'ground']:
                        # Sample power/ground nets - important for decoupling
                        sampled_conns = self._sample_power_net_connections(
                            unique_refs, base_weight, power_net_sample_size
                        )
                        self.connections.extend(sampled_conns)
                        sampled_power_nets += 1
                        logger.debug(f"Sampled {len(sampled_conns)} connections from {net_name}")
                    else:
                        # Skip large signal nets entirely
                        logger.debug(f"Skipping large signal net {net_name} ({len(pads)} pads)")
                    continue
                
                # Create pairwise connections for small nets
                for i in range(len(unique_refs)):
                    for j in range(i + 1, len(unique_refs)):
                        weight = base_weight
                        ref1, ref2 = unique_refs[i], unique_refs[j]
                        self.connections.append((ref1, ref2, weight))
            
            logger.info(f"Parsed {len(self.connections)} connections from {len(nets)} nets")
            if sampled_power_nets > 0:
                logger.info(f"  Sampled {sampled_power_nets} large power/ground nets")
            logger.info(f"  Components with nets: {len(self.component_refs)}")
            
        except Exception as e:
            logger.error(f"Error parsing netlist: {e}")
            traceback.print_exc()
        
        return self.connections
    
    def get_connections_for_components(self, component_refs: Set[str]) -> List[Tuple[str, str, float]]:
        """Filter connections to only include specified components."""
        return [
            (ref1, ref2, weight)
            for ref1, ref2, weight in self.connections
            if ref1 in component_refs and ref2 in component_refs
        ]


# ============================================================================
# BASIC PLACEMENT SANITY CHECK (Not full DRC - limited to simple checks)
# ============================================================================

class BasicPlacementCheck:
    """Basic sanity check for component placement.
    
    IMPORTANT: This is NOT a full KiCad DRC. It only checks:
    - Components outside board bounding box
    - Bounding box overlaps (not courtyard-based)
    
    For full DRC, use KiCad's native tools after placement.
    """
    
    def __init__(self, board):
        self.board = board
        self.violations: List[dict] = []
    
    @classmethod
    def from_updater(cls, updater: 'KiCadPCBUpdater') -> Optional['BasicPlacementCheck']:
        """Create checker from KiCadPCBUpdater."""
        if not updater.use_pcbnew or updater.board is None:
            logger.warning("Placement check requires pcbnew API - skipping")
            return None
        return cls(updater.board)
    
    def run(self) -> Tuple[int, int]:
        """Run DRC and return (error_count, warning_count).
        
        Note: Full DRC requires KiCad 8.0+. Earlier versions have limited API.
        """
        if self.board is None:
            return (-1, -1)
        
        self.violations.clear()
        errors = 0
        warnings = 0
        
        try:
            # Check for basic placement issues we can detect
            
            # 1. Components outside board area
            board_bb = self.board.GetBoardEdgesBoundingBox()
            if board_bb.GetWidth() > 0:
                for fp in self.board.GetFootprints():
                    fp_bb = fp.GetBoundingBox()
                    if not board_bb.Contains(fp_bb):
                        ref = fp.GetReference()
                        self.violations.append({
                            'type': 'outside_board',
                            'ref': ref,
                            'severity': 'warning'
                        })
                        warnings += 1
            
            # 2. Overlapping footprints (courtyard check)
            footprints = list(self.board.GetFootprints())
            for i in range(len(footprints)):
                for j in range(i + 1, len(footprints)):
                    fp1, fp2 = footprints[i], footprints[j]
                    
                    # Check if bounding boxes overlap (simple check)
                    bb1 = fp1.GetBoundingBox()
                    bb2 = fp2.GetBoundingBox()
                    
                    if bb1.Intersects(bb2):
                        # Check if on same layer
                        if fp1.GetLayer() == fp2.GetLayer():
                            ref1 = fp1.GetReference()
                            ref2 = fp2.GetReference()
                            self.violations.append({
                                'type': 'footprint_overlap',
                                'refs': (ref1, ref2),
                                'severity': 'error'
                            })
                            errors += 1
            
            # Log results
            if errors > 0 or warnings > 0:
                logger.warning(f"DRC: {errors} errors, {warnings} warnings")
                for v in self.violations[:5]:  # Show first 5
                    if v['type'] == 'outside_board':
                        logger.warning(f"  {v['ref']}: Outside board boundary")
                    elif v['type'] == 'footprint_overlap':
                        logger.warning(f"  {v['refs'][0]} overlaps {v['refs'][1]}")
            else:
                logger.info("DRC: No issues detected")
            
        except Exception as e:
            logger.error(f"DRC check failed: {e}")
            return (-1, -1)
        
        return (errors, warnings)
    
    def get_violations(self) -> List[dict]:
        """Get list of DRC violations."""
        return self.violations


# ============================================================================
# KICAD PCB UPDATER (with pcbnew API when available)
# ============================================================================

class KiCadPCBUpdater:
    """Updates KiCad PCB file with new component positions.
    
    Uses pcbnew API when available (robust), falls back to regex (fragile).
    """
    
    def __init__(self, pcb_path: str):
        self.pcb_path = pcb_path
        self.board = None
        self.use_pcbnew = PCBNEW_AVAILABLE
        
        if self.use_pcbnew:
            self._load_with_pcbnew()
        else:
            self._load_with_regex()
    
    def _load_with_pcbnew(self):
        """Load PCB using KiCad's Python API."""
        try:
            self.board = pcbnew.LoadBoard(self.pcb_path)
            logger.info(f"Loaded PCB with pcbnew: {self.pcb_path}")
        except Exception as e:
            logger.error(f"Failed to load with pcbnew: {e}")
            logger.info("Falling back to regex parsing")
            self.use_pcbnew = False
            self._load_with_regex()
    
    def _load_with_regex(self):
        """Fallback: Load PCB as text for regex manipulation."""
        with open(self.pcb_path, 'r') as f:
            self.content = f.read()
        logger.info(f"Loaded PCB with regex parser: {self.pcb_path}")
    
    def update_footprint_position(self, ref: str, x: float, y: float,
                                   rotation: float = None,
                                   layer: str = None) -> bool:
        """Update component position and optionally layer."""
        if self.use_pcbnew:
            return self._update_with_pcbnew(ref, x, y, rotation, layer)
        else:
            return self._update_with_regex(ref, x, y, rotation, layer)

    def _update_with_pcbnew(self, ref: str, x: float, y: float,
                             rotation: float = None,
                             layer: str = None) -> bool:
        """Update position using pcbnew API (robust)."""
        try:
            fp = self.board.FindFootprintByReference(ref)
            if fp is None:
                logger.warning(f"Footprint not found: {ref}")
                return False

            # Convert mm to native units (nanometers in KiCad 7+)
            # Use pcbnew.FromMM() for proper conversion
            pos = pcbnew.VECTOR2I(int(pcbnew.FromMM(x)), int(pcbnew.FromMM(y)))
            fp.SetPosition(pos)

            if rotation is not None:
                # KiCad uses decidegrees (1/10 degree)
                fp.SetOrientationDegrees(rotation)

            if layer is not None:
                # Set footprint layer (front = F.Cu, back = B.Cu)
                if layer.lower() == "back":
                    fp.SetLayerAndFlip(pcbnew.B_Cu)
                    logger.debug(f"  {ref}: Flipped to back layer")
                elif layer.lower() == "front":
                    fp.SetLayerAndFlip(pcbnew.F_Cu)

            return True
        except Exception as e:
            logger.error(f"Error updating {ref}: {e}")
            return False
    
    def _update_with_regex(self, ref: str, x: float, y: float,
                            rotation: float = None,
                            layer: str = None) -> bool:
        """Fallback: Update position using regex (fragile).

        Note: Layer changes are not supported in regex mode.
        """
        if layer is not None:
            logger.warning(f"  {ref}: Layer change requires pcbnew API (ignored)")

        ref_pattern = rf'\(property "Reference" "{re.escape(ref)}"'
        ref_matches = list(re.finditer(ref_pattern, self.content))
        
        if not ref_matches:
            return False
        
        for ref_match in ref_matches:
            search_start = max(0, ref_match.start() - 5000)
            search_region = self.content[search_start:ref_match.start()]
            
            fp_matches = list(re.finditer(r'\(footprint "', search_region))
            if not fp_matches:
                continue
            
            fp_start_in_region = fp_matches[-1].start()
            fp_absolute_start = search_start + fp_start_in_region
            
            fp_header = self.content[fp_absolute_start:fp_absolute_start + 500]
            
            at_pattern = r'\(at\s+([0-9.-]+)\s+([0-9.-]+)(\s+[0-9.-]+)?\)'
            at_match = re.search(at_pattern, fp_header)
            
            if at_match:
                if rotation is not None:
                    new_at = f"(at {x} {y} {rotation})"
                elif at_match.group(3):
                    new_at = f"(at {x} {y}{at_match.group(3)})"
                else:
                    new_at = f"(at {x} {y})"
                
                at_absolute_start = fp_absolute_start + at_match.start()
                at_absolute_end = fp_absolute_start + at_match.end()
                self.content = (self.content[:at_absolute_start] + new_at + 
                               self.content[at_absolute_end:])
                return True
        
        return False
    
    def save(self, output_path: str):
        """Save the modified PCB."""
        if self.use_pcbnew:
            try:
                self.board.Save(output_path)
                logger.info(f"Saved with pcbnew: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save with pcbnew: {e}")
        else:
            with open(output_path, 'w') as f:
                f.write(self.content)
            logger.info(f"Saved with regex: {output_path}")


# ============================================================================
# COMPONENT SIZE EXTRACTION
# ============================================================================

class ComponentSizeExtractor:
    """Extract actual component sizes from KiCad PCB file.

    Uses pcbnew API to get accurate footprint dimensions:
    - GetFpPadsLocalBbox(): Pad-only bounding box (tightest fit)
    - GetCourtyard(): Courtyard polygon (includes clearance)
    - GetBoundingBox(): Full bounding box (includes silkscreen, too large)

    Per IPC-7351C / KiCad KLC F5.3:
    - Standard courtyard clearance: 0.25mm
    - Parts ≤0603: 0.15mm clearance
    - Connectors: 0.5mm clearance
    - BGAs: 1.0mm clearance
    """

    # Clearance based on package size (mm)
    CLEARANCE_SMALL = 0.15  # 0402, 0201
    CLEARANCE_STANDARD = 0.25  # Most parts
    CLEARANCE_CONNECTOR = 0.5
    CLEARANCE_BGA = 1.0

    def __init__(self, pcb_path: str):
        self.pcb_path = pcb_path
        self.board = None
        self.sizes: Dict[str, Tuple[float, float]] = {}
        self._load_pcb()

    def _load_pcb(self):
        """Load PCB using pcbnew API."""
        if not PCBNEW_AVAILABLE:
            logger.warning("pcbnew not available - cannot extract component sizes")
            return

        try:
            self.board = pcbnew.LoadBoard(self.pcb_path)
            logger.info(f"Loaded PCB for size extraction: {self.pcb_path}")
        except Exception as e:
            logger.error(f"Failed to load PCB: {e}")

    def _get_clearance_for_ref(self, ref: str) -> float:
        """Determine appropriate clearance based on reference designator."""
        # Connectors
        if ref.startswith('J') or ref.startswith('P'):
            return self.CLEARANCE_CONNECTOR
        # BGAs (usually U* with large pin count)
        # For now, use standard clearance for ICs
        return self.CLEARANCE_STANDARD

    def extract_sizes(self, use_courtyard: bool = False,
                      add_clearance: bool = True) -> Dict[str, Tuple[float, float]]:
        """Extract component sizes from the PCB.

        Args:
            use_courtyard: If True, use courtyard bounds. If False, use pad bounds.
            add_clearance: If True, add appropriate clearance margin.

        Returns:
            Dict mapping reference designator to (width, height) in mm.
        """
        if not self.board:
            return {}

        for fp in self.board.GetFootprints():
            ref = fp.GetReference()

            try:
                if use_courtyard:
                    # Try courtyard first
                    fp.BuildCourtyardCaches()
                    layer = pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
                    courtyard = fp.GetCourtyard(layer)
                    if not courtyard.IsEmpty():
                        bbox = courtyard.BBox()
                    else:
                        # Fallback to pad bbox
                        bbox = fp.GetFpPadsLocalBbox()
                else:
                    # Use pad-only bounding box (tightest fit)
                    bbox = fp.GetFpPadsLocalBbox()

                # Convert from KiCad internal units (nm) to mm
                width = pcbnew.ToMM(bbox.GetWidth())
                height = pcbnew.ToMM(bbox.GetHeight())

                # Add clearance if requested
                if add_clearance:
                    clearance = self._get_clearance_for_ref(ref)
                    width += 2 * clearance
                    height += 2 * clearance

                # Ensure minimum size (some footprints report 0)
                width = max(width, 0.5)
                height = max(height, 0.5)

                self.sizes[ref] = (width, height)
                logger.debug(f"  {ref}: {width:.2f} x {height:.2f} mm")

            except Exception as e:
                logger.warning(f"  {ref}: Failed to extract size: {e}")
                # Use default fallback
                self.sizes[ref] = get_component_size(ref)

        logger.info(f"Extracted sizes for {len(self.sizes)} components")
        return self.sizes

    def compare_with_config(self, config_sizes: Dict[str, Tuple[float, float]]) -> None:
        """Compare extracted sizes with config sizes and log differences."""
        logger.info("\n=== Component Size Comparison ===")
        logger.info(f"{'Ref':<8} {'Config':>12} {'Actual':>12} {'Diff':>8}")
        logger.info("-" * 44)

        for ref, (cw, ch) in config_sizes.items():
            if ref in self.sizes:
                aw, ah = self.sizes[ref]
                config_area = cw * ch
                actual_area = aw * ah
                diff_pct = ((config_area - actual_area) / actual_area) * 100 if actual_area > 0 else 0

                if abs(diff_pct) > 20:  # Flag significant differences
                    flag = " ***" if diff_pct > 50 else " *"
                else:
                    flag = ""

                logger.info(f"{ref:<8} {cw:.1f}x{ch:.1f}  {aw:.1f}x{ah:.1f}  {diff_pct:+.0f}%{flag}")


# ============================================================================
# PLACEMENT ENGINE
# ============================================================================

class PlacementEngine:
    """Main engine for component placement."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.groups: Dict[str, ComponentGroup] = {}
        self.components: Dict[str, Component] = {}
        self.connections: List[Tuple[str, str, float]] = []
        
        self._parse_config()
    
    def _load_config(self) -> dict:
        """Load and validate YAML configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigValidationError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML: {e}")
        
        validate_config(config)
        return config
    
    def _parse_config(self):
        """Parse configuration into components and groups."""
        board = self.config.get('board', {})
        self.board_width = board.get('width', 50.0)
        self.board_height = board.get('height', 45.0)
        
        fd_config = self.config.get('force_directed', {})
        self.fd_config = ForceDirectedConfig(
            enabled=fd_config.get('enabled', True),
            iterations=fd_config.get('iterations', 200),
            attraction_strength=fd_config.get('attraction_strength', 0.15),
            repulsion_strength=fd_config.get('repulsion_strength', 500.0),
            damping=fd_config.get('damping', 0.82),
            collision_strength=fd_config.get('collision_strength', 50.0),
            min_clearance=fd_config.get('min_clearance', 0.5),
            random_seed=fd_config.get('random_seed', 42),
            max_velocity=fd_config.get('max_velocity', 3.0),
        )
        
        sizes_config = self.config.get('component_sizes', {})
        for ref, size in sizes_config.items():
            if isinstance(size, list) and len(size) == 2:
                COMPONENT_SIZES[ref] = tuple(size)
        
        groups_config = self.config.get('groups', {})
        for group_name, group_data in groups_config.items():
            group = ComponentGroup(
                name=group_name,
                anchor=group_data.get('anchor', ''),
                position=tuple(group_data.get('position', [0.0, 0.0])),
                layer=group_data.get('layer', 'front'),
                components=group_data.get('components', {}),
            )
            self.groups[group_name] = group
            
            for ref, comp_data in group.components.items():
                offset = comp_data.get('offset', [0.0, 0.0])
                size = comp_data.get('size', get_component_size(ref))
                comp_layer = comp_data.get('layer', group.layer)
                
                component = Component(
                    ref=ref,
                    x=group.position[0] + offset[0],
                    y=group.position[1] + offset[1],
                    rotation=comp_data.get('rotation', 0),
                    width=size[0] if isinstance(size, (list, tuple)) else size,
                    height=size[1] if isinstance(size, (list, tuple)) else size,
                    group=group_name,
                    layer=comp_layer,
                    fixed=(ref == group.anchor),
                )
                self.components[ref] = component
        
        self.connections = self.config.get('connections', [])
        
        logger.info(f"Parsed {len(self.groups)} groups, {len(self.components)} components")
    
    def optimize(self) -> Dict[str, Component]:
        """Run force-directed optimization."""
        placer = ForceDirectedPlacer(self.fd_config, self.board_width, self.board_height)
        
        for comp in self.components.values():
            placer.add_component(comp)
        
        for conn in self.connections:
            if len(conn) >= 2:
                weight = conn[2] if len(conn) > 2 else 1.0
                placer.add_connection(conn[0], conn[1], weight)
        
        return placer.run()
    
    def resolve_overlaps(self, max_iterations: int = 50):
        """Resolve remaining overlaps."""
        logger.info("\n=== Resolving Remaining Overlaps ===")
        
        for iteration in range(max_iterations):
            overlaps = []
            refs = list(self.components.keys())
            
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    c1, c2 = self.components[refs[i]], self.components[refs[j]]
                    if c1.overlaps(c2, margin=0):
                        overlaps.append((refs[i], refs[j]))
            
            if not overlaps:
                logger.info(f"  All overlaps resolved after {iteration} iterations")
                return
            
            for ref1, ref2 in overlaps:
                c1, c2 = self.components[ref1], self.components[ref2]
                
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 0.1:
                    angle = random.uniform(0, 2 * math.pi)
                    dx, dy = math.cos(angle), math.sin(angle)
                    dist = 1.0
                
                min_sep = (c1.width + c2.width) / 2 + self.fd_config.min_clearance
                push = (min_sep - dist) / 2 + 0.5
                
                if not c1.fixed:
                    c1.x += (dx / dist) * push
                    c1.y += (dy / dist) * push
                if not c2.fixed:
                    c2.x -= (dx / dist) * push
                    c2.y -= (dy / dist) * push
        
        remaining = sum(1 for i in range(len(refs)) 
                       for j in range(i+1, len(refs)) 
                       if self.components[refs[i]].overlaps(self.components[refs[j]], margin=0))
        if remaining > 0:
            logger.warning(f"  {remaining} overlaps remain after {max_iterations} iterations")


# ============================================================================
# PLACEMENT VISUALIZER
# ============================================================================

class PlacementVisualizer:
    """Matplotlib-based placement visualization for before/after comparison."""
    
    def __init__(self, board_width: float, board_height: float):
        self.board_width = board_width
        self.board_height = board_height
        self.before_positions: Dict[str, Tuple[float, float, float, float, float]] = {}  # ref: (x, y, w, h, rot)
        self.after_positions: Dict[str, Tuple[float, float, float, float, float]] = {}
    
    def capture_before(self, components: Dict[str, Component]):
        """Capture component positions before optimization."""
        for ref, comp in components.items():
            self.before_positions[ref] = (comp.x, comp.y, comp.width, comp.height, comp.rotation)
    
    def capture_after(self, components: Dict[str, Component]):
        """Capture component positions after optimization."""
        for ref, comp in components.items():
            self.after_positions[ref] = (comp.x, comp.y, comp.width, comp.height, comp.rotation)
    
    def plot(self, output_path: str = None):
        """Generate before/after comparison plot."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - skipping visualization")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Color coding
        colors = {
            'U': '#FF6B6B',   # ICs - red
            'C': '#4ECDC4',   # Capacitors - teal
            'R': '#45B7D1',   # Resistors - blue
            'L': '#96CEB4',   # Inductors - green
            'B': '#FFEAA7',   # Battery - yellow
            'AN': '#DDA0DD', # Antenna - plum
            'H': '#FFB347',   # Headers - orange
            'LE': '#98D8C8', # LED - mint
        }
        
        def get_color(ref: str) -> str:
            for prefix, color in colors.items():
                if ref.startswith(prefix):
                    return color
            return '#CCCCCC'
        
        def draw_components(ax, positions: Dict, title: str):
            import matplotlib.transforms as transforms

            ax.set_xlim(-self.board_width/2 - 2, self.board_width/2 + 2)
            ax.set_ylim(-self.board_height/2 - 2, self.board_height/2 + 2)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')

            # Draw board outline
            board_rect = patches.Rectangle(
                (-self.board_width/2, -self.board_height/2),
                self.board_width, self.board_height,
                linewidth=2, edgecolor='black', facecolor='#F5F5DC', alpha=0.3
            )
            ax.add_patch(board_rect)

            # Draw components
            for ref, (x, y, w, h, rot) in positions.items():
                color = get_color(ref)

                # Create rectangle with rotation around center
                # matplotlib Rectangle rotates around corner, so we use a transform
                rect = patches.Rectangle(
                    (-w/2, -h/2), w, h,
                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
                )
                # Apply rotation around center, then translate to position
                t = transforms.Affine2D().rotate_deg(rot).translate(x, y) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)

                # Add label for larger components
                if w > 3 or h > 3:
                    ax.annotate(ref, (x, y), ha='center', va='center', fontsize=7,
                               fontweight='bold', color='black')

            # Draw grid
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='gray', linewidth=0.5)
            ax.axvline(x=0, color='gray', linewidth=0.5)
        
        draw_components(axes[0], self.before_positions, 'BEFORE Optimization')
        draw_components(axes[1], self.after_positions, 'AFTER Optimization')
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor=colors['U'], label='ICs'),
            patches.Patch(facecolor=colors['C'], label='Capacitors'),
            patches.Patch(facecolor=colors['R'], label='Resistors'),
            patches.Patch(facecolor=colors['L'], label='Inductors'),
            patches.Patch(facecolor=colors['B'], label='Battery'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
                  bbox_to_anchor=(0.5, 0.02))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization: {output_path}")
        else:
            plt.show()
        
        plt.close()
        return output_path


# ============================================================================
# MAIN
# ============================================================================

def find_latest_pcb(build_dir: str) -> Optional[str]:
    """Find the latest KiCad PCB file."""
    pattern = os.path.join(build_dir, "*.kicad_pcb")
    files = glob_module.glob(pattern)
    files = [f for f in files if not f.endswith("_placed.kicad_pcb")]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description='KiCad PCB Auto-Placer v2.0 (with pcbnew API)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=YAML_SCHEMA
    )
    parser.add_argument('--config', default='placement_config.yaml',
                        help='Path to YAML configuration')
    parser.add_argument('--pcb', help='Path to input KiCad PCB file')
    parser.add_argument('--output', help='Path to output KiCad PCB file')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='Enable force-directed optimization (default)')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                        help='Disable optimization')
    parser.add_argument('--no-rotate', action='store_true',
                        help='Disable auto-rotation optimization')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate before/after placement visualization')
    parser.add_argument('--parse-netlist', action='store_true',
                        help='Auto-extract connections from PCB netlist (requires pcbnew)')
    parser.add_argument('--drc', action='store_true',
                        help='Run design rule check after placement')
    parser.add_argument('--auto-size', action='store_true',
                        help='Auto-detect component sizes from KiCad (requires pcbnew)')
    parser.add_argument('--compare-sizes', action='store_true',
                        help='Compare config sizes with actual KiCad sizes')
    parser.add_argument('--use-courtyard', action='store_true',
                        help='Use courtyard bounds instead of pad bounds for sizing')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_dir, args.config)
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading configuration: {config_path}")
    
    try:
        engine = PlacementEngine(config_path)
    except ConfigValidationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    if args.seed:
        engine.fd_config.random_seed = args.seed
    
    if args.no_rotate:
        engine.fd_config.auto_rotate = False
    
    engine.fd_config.enabled = args.optimize
    
    # Find PCB file early (needed for netlist parsing)
    if args.pcb:
        pcb_path = args.pcb
    else:
        build_dir = os.path.join(project_dir, "build/builds/default")
        pcb_path = find_latest_pcb(build_dir)
    
    if not pcb_path or not os.path.exists(pcb_path):
        logger.error("PCB file not found")
        sys.exit(1)

    # Auto-detect component sizes from KiCad
    if args.auto_size or args.compare_sizes:
        logger.info("\n=== Extracting Component Sizes from KiCad ===")
        size_extractor = ComponentSizeExtractor(pcb_path)
        extracted_sizes = size_extractor.extract_sizes(
            use_courtyard=args.use_courtyard,
            add_clearance=True
        )

        if args.compare_sizes:
            # Build config sizes dict for comparison
            config_sizes = {ref: (c.width, c.height) for ref, c in engine.components.items()}
            size_extractor.compare_with_config(config_sizes)

        if args.auto_size and extracted_sizes:
            # Update component sizes with extracted values
            updated_count = 0
            for ref, (w, h) in extracted_sizes.items():
                if ref in engine.components:
                    old_w, old_h = engine.components[ref].width, engine.components[ref].height
                    engine.components[ref].width = w
                    engine.components[ref].height = h
                    if abs(old_w - w) > 0.1 or abs(old_h - h) > 0.1:
                        logger.debug(f"  {ref}: {old_w:.1f}x{old_h:.1f} -> {w:.1f}x{h:.1f}")
                        updated_count += 1
            logger.info(f"Updated {updated_count} component sizes from KiCad")

    # Parse netlist for auto-connections
    netlist_connections = []
    if args.parse_netlist:
        logger.info("\n=== Parsing Netlist ===")
        netlist_parser = NetlistParser.from_pcb_file(pcb_path)
        if netlist_parser:
            netlist_connections = netlist_parser.parse(max_pins_per_net=10)
            # Add to engine's connections (merged with YAML)
            component_refs = set(engine.components.keys())
            filtered_conns = netlist_parser.get_connections_for_components(component_refs)
            logger.info(f"  Using {len(filtered_conns)} connections for placed components")
            engine.connections.extend(filtered_conns)
    
    # Setup visualization
    visualizer = None
    if args.visualize:
        if MATPLOTLIB_AVAILABLE:
            visualizer = PlacementVisualizer(engine.board_width, engine.board_height)
            visualizer.capture_before(engine.components)
        else:
            logger.warning("matplotlib not available - install with: pip install matplotlib")
    
    if args.optimize:
        logger.info("\n=== Force-Directed Optimization ===")
        engine.optimize()
    
    engine.resolve_overlaps()
    
    # Capture after positions for visualization
    if visualizer:
        visualizer.capture_after(engine.components)
    
    logger.info(f"\nUpdating PCB: {pcb_path}")
    updater = KiCadPCBUpdater(pcb_path)
    
    updated = 0
    not_found = []
    
    for ref, comp in engine.components.items():
        if updater.update_footprint_position(ref, comp.x, comp.y, comp.rotation, comp.layer):
            layer_info = f" [{comp.layer}]" if comp.layer == "back" else ""
            logger.info(f"  ✓ {ref}: ({comp.x:.1f}, {comp.y:.1f}, {comp.rotation}°){layer_info}")
            updated += 1
        else:
            not_found.append(ref)
    
    if args.output:
        output_path = args.output
    else:
        output_path = pcb_path.replace('.kicad_pcb', '_placed.kicad_pcb')
    
    updater.save(output_path)
    
    # Run DRC if requested
    drc_errors, drc_warnings = 0, 0
    if args.drc:
        logger.info("\n=== Design Rule Check ===")
        drc_checker = BasicPlacementCheck.from_updater(updater)
        if drc_checker:
            drc_errors, drc_warnings = drc_checker.run()
    
    # Generate visualization
    viz_path = None
    if visualizer:
        viz_path = output_path.replace('.kicad_pcb', '_placement.png')
        visualizer.plot(viz_path)
    
    print(f"\n{'='*50}")
    print(f"Placement Summary")
    print(f"{'='*50}")
    print(f"API: {'pcbnew (robust)' if updater.use_pcbnew else 'regex (fallback)'}")
    print(f"Auto-rotation: {'enabled' if engine.fd_config.auto_rotate else 'disabled'}")
    print(f"Netlist parsing: {'enabled' if args.parse_netlist else 'disabled'}")
    print(f"Groups: {len(engine.groups)}")
    print(f"Components updated: {updated}")
    print(f"Components not found: {len(not_found)}")
    if not_found:
        print(f"  Missing: {', '.join(not_found[:5])}{'...' if len(not_found) > 5 else ''}")
    if args.drc:
        print(f"DRC: {drc_errors} errors, {drc_warnings} warnings")
    print(f"Output: {output_path}")
    if viz_path:
        print(f"Visualization: {viz_path}")


if __name__ == "__main__":
    main()

