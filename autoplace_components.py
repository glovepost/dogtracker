#!/usr/bin/env python3
"""
Advanced KiCad PCB Component Auto-Placer with Collision Detection

Features:
- YAML configuration for easy editing
- Component grouping with relative positioning
- Force-directed algorithm for initial placement optimization
- Bounding box collision detection (no courtyard data needed)
- Overlap resolution with push-apart forces

Usage:
    python autoplace_components.py [--config placement_config.yaml] [--optimize]
"""

import re
import os
import sys
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    import yaml
except ImportError:
    print("Installing pyyaml...")
    os.system(f"{sys.executable} -m pip install pyyaml")
    import yaml


# ============================================================================
# COMPONENT SIZE DATABASE
# Since atopile footprints lack courtyards, we define bounding boxes manually.
# Format: (width_mm, height_mm) - includes clearance margin
# ============================================================================
COMPONENT_SIZES = {
    # ICs and Modules
    "U1": (14.0, 14.0),    # RAK3172-SIP (12x12 + 1mm margin each side)
    "U2": (4.5, 4.5),      # LIS3DHTR LGA-16 (3x3 + margins)
    "U3": (4.0, 2.5),      # FT24C02A SOT-23-5
    "U4": (4.5, 4.0),      # QMI8658C LGA-14 (3x2.5 + margins)
    "U5": (4.5, 4.5),      # HDC2080 WSON-6 (3x3 + margins)
    
    # Battery holder
    "B1": (26.0, 26.0),    # CR2032 holder (24mm diameter + margin)
    
    # Antenna
    "ANT1": (10.0, 5.0),   # SMD antenna
    
    # Connectors
    "H1": (15.0, 4.0),     # SWD 5-pin header
    
    # LED
    "LED1": (2.5, 1.5),    # 0603 LED
    
    # Default sizes for passives by prefix
    "DEFAULT_R": (1.8, 1.2),   # 0402 resistor with margin
    "DEFAULT_C": (1.8, 1.2),   # 0402 capacitor with margin
    "DEFAULT_L": (2.5, 2.0),   # 0603/0805 inductor with margin
}

def get_component_size(ref: str) -> Tuple[float, float]:
    """Get the bounding box size for a component reference."""
    # Check for explicit size
    if ref in COMPONENT_SIZES:
        return COMPONENT_SIZES[ref]
    
    # Determine by prefix
    prefix = ''.join(c for c in ref if c.isalpha())
    
    if prefix == 'R':
        return COMPONENT_SIZES["DEFAULT_R"]
    elif prefix == 'C':
        return COMPONENT_SIZES["DEFAULT_C"]
    elif prefix == 'L':
        return COMPONENT_SIZES["DEFAULT_L"]
    else:
        # Default fallback
        return (3.0, 3.0)


@dataclass
class Component:
    """Represents a component with position, size, and properties."""
    ref: str
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    width: float = 2.0   # Bounding box width
    height: float = 2.0  # Bounding box height
    group: Optional[str] = None
    note: str = ""
    vx: float = 0.0      # Velocity for force-directed
    vy: float = 0.0
    fixed: bool = False  # Don't move during optimization
    layer: str = "front" # front or back - only same-layer components collide

    
    def __post_init__(self):
        """Initialize size from database if not set."""
        w, h = get_component_size(self.ref)
        if self.width == 2.0:  # Default value, needs update
            self.width = w
        if self.height == 2.0:
            self.height = h
    
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
        """Check if this component overlaps with another (with margin).
        Only components on the same layer can overlap."""
        # Different layers don't collide
        if self.layer != other.layer:
            return False
        return (
            self.left - margin < other.right + margin and
            self.right + margin > other.left - margin and
            self.bottom - margin < other.top + margin and
            self.top + margin > other.bottom - margin
        )
    
    def overlap_amount(self, other: 'Component') -> Tuple[float, float]:
        """Calculate overlap amount in x and y directions."""
        if not self.overlaps(other, margin=0):
            return 0.0, 0.0
        
        # Calculate overlap in each direction
        overlap_x = min(self.right, other.right) - max(self.left, other.left)
        overlap_y = min(self.top, other.top) - max(self.bottom, other.bottom)
        
        return max(0, overlap_x), max(0, overlap_y)


@dataclass
class ComponentGroup:
    """A group of components that maintain relative positions."""
    name: str
    anchor: str
    position: Tuple[float, float]
    components: Dict[str, dict]


@dataclass
class ForceDirectedConfig:
    """Configuration for force-directed placement algorithm."""
    enabled: bool = True
    iterations: int = 100
    attraction_strength: float = 0.1
    repulsion_strength: float = 500.0
    damping: float = 0.85
    collision_strength: float = 50.0  # Force to push apart overlapping components
    min_clearance: float = 0.5        # Minimum clearance between components


class ForceDirectedPlacer:
    """
    Force-directed graph-based component placement algorithm with collision detection.
    
    Components connected by nets attract each other.
    All components repel to avoid overlap.
    Overlapping components experience strong push-apart forces.
    """
    
    def __init__(self, config: ForceDirectedConfig, board_width: float, board_height: float):
        self.config = config
        self.board_width = board_width
        self.board_height = board_height
        self.components: Dict[str, Component] = {}
        self.connections: List[Tuple[str, str, float]] = []
    
    def add_component(self, component: Component):
        """Add a component to the placer."""
        self.components[component.ref] = component
    
    def add_connection(self, comp1: str, comp2: str, weight: float = 1.0):
        """Add a netlist connection between two components."""
        if comp1 in self.components and comp2 in self.components:
            self.connections.append((comp1, comp2, weight))
    
    def calculate_collision_force(self, c1: Component, c2: Component) -> Tuple[float, float]:
        """Calculate collision avoidance force between overlapping components."""
        if not c1.overlaps(c2, margin=self.config.min_clearance):
            return 0.0, 0.0
        
        # Get direction from c2 to c1
        dx = c1.x - c2.x
        dy = c1.y - c2.y
        
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 0.1:
            # Components at same position, random push
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle)
            dy = math.sin(angle)
            distance = 1.0
        
        # Calculate overlap amount
        overlap_x, overlap_y = c1.overlap_amount(c2)
        overlap = max(overlap_x, overlap_y) + self.config.min_clearance
        
        # Strong force proportional to overlap
        force = self.config.collision_strength * overlap
        
        # Apply in direction away from other component
        fx = (dx / distance) * force
        fy = (dy / distance) * force
        
        return fx, fy
    
    def calculate_repulsion_force(self, c1: Component, c2: Component) -> Tuple[float, float]:
        """Calculate repulsion force between two components (distance-based)."""
        dx = c1.x - c2.x
        dy = c1.y - c2.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Minimum distance based on component sizes
        min_dist = (c1.width + c2.width) / 2 + (c1.height + c2.height) / 2
        min_dist = min_dist / 2 + self.config.min_clearance
        
        if distance < 0.1:
            distance = 0.1
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)
        
        # Coulomb's law with size-aware minimum distance
        if distance < min_dist:
            # Stronger repulsion when closer than minimum
            force = self.config.repulsion_strength * (min_dist / distance)**2
        else:
            force = self.config.repulsion_strength / (distance * distance)
        
        # Normalize and apply force
        fx = (dx / distance) * force
        fy = (dy / distance) * force
        
        return fx, fy
    
    def calculate_attraction_force(self, c1: Component, c2: Component, weight: float) -> Tuple[float, float]:
        """Calculate attraction force between connected components."""
        dx = c2.x - c1.x
        dy = c2.y - c1.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 0.1:
            return 0.0, 0.0
        
        # Hooke's law: F = k * d
        force = self.config.attraction_strength * distance * weight
        
        # Normalize and apply force
        fx = (dx / distance) * force
        fy = (dy / distance) * force
        
        return fx, fy
    
    def constrain_to_board(self, component: Component):
        """Keep component within board boundaries."""
        margin = component.width / 2 + 2.0  # Component size + edge margin
        margin_y = component.height / 2 + 2.0
        
        half_w = self.board_width / 2 - margin
        half_h = self.board_height / 2 - margin_y
        
        component.x = max(-half_w, min(half_w, component.x))
        component.y = max(-half_h, min(half_h, component.y))
    
    def count_overlaps(self) -> int:
        """Count the number of overlapping component pairs."""
        overlaps = 0
        refs = list(self.components.keys())
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                c1 = self.components[refs[i]]
                c2 = self.components[refs[j]]
                if c1.overlaps(c2, margin=0):
                    overlaps += 1
        return overlaps
    
    def run(self) -> Dict[str, Component]:
        """Run the force-directed placement algorithm with collision detection."""
        if not self.config.enabled:
            return self.components
        
        print(f"Running force-directed placement ({self.config.iterations} iterations)...")
        print(f"  Initial overlaps: {self.count_overlaps()}")
        
        for iteration in range(self.config.iterations):
            # Calculate forces for each component
            forces: Dict[str, Tuple[float, float]] = {ref: (0.0, 0.0) for ref in self.components}
            
            refs = list(self.components.keys())
            
            # Process all component pairs
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    c1 = self.components[refs[i]]
                    c2 = self.components[refs[j]]
                    
                    # Collision forces (highest priority)
                    cfx, cfy = self.calculate_collision_force(c1, c2)
                    if cfx != 0 or cfy != 0:
                        if not c1.fixed:
                            forces[refs[i]] = (forces[refs[i]][0] + cfx, forces[refs[i]][1] + cfy)
                        if not c2.fixed:
                            forces[refs[j]] = (forces[refs[j]][0] - cfx, forces[refs[j]][1] - cfy)
                    else:
                        # Normal repulsion forces when not overlapping
                        fx, fy = self.calculate_repulsion_force(c1, c2)
                        if not c1.fixed:
                            forces[refs[i]] = (forces[refs[i]][0] + fx, forces[refs[i]][1] + fy)
                        if not c2.fixed:
                            forces[refs[j]] = (forces[refs[j]][0] - fx, forces[refs[j]][1] - fy)
            
            # Attraction forces for connected components
            for comp1, comp2, weight in self.connections:
                if comp1 not in self.components or comp2 not in self.components:
                    continue
                    
                c1 = self.components[comp1]
                c2 = self.components[comp2]
                
                # Don't attract if overlapping
                if c1.overlaps(c2, margin=self.config.min_clearance):
                    continue
                
                fx, fy = self.calculate_attraction_force(c1, c2, weight)
                
                if not c1.fixed:
                    forces[comp1] = (forces[comp1][0] + fx, forces[comp1][1] + fy)
                if not c2.fixed:
                    forces[comp2] = (forces[comp2][0] - fx, forces[comp2][1] - fy)
            
            # Apply forces with velocity and damping
            for ref, (fx, fy) in forces.items():
                component = self.components[ref]
                if component.fixed:
                    continue
                
                # Update velocity
                component.vx = (component.vx + fx) * self.config.damping
                component.vy = (component.vy + fy) * self.config.damping
                
                # Cap velocity
                max_velocity = 3.0
                speed = math.sqrt(component.vx**2 + component.vy**2)
                if speed > max_velocity:
                    component.vx = (component.vx / speed) * max_velocity
                    component.vy = (component.vy / speed) * max_velocity
                
                # Update position
                component.x += component.vx
                component.y += component.vy
                
                # Constrain to board
                self.constrain_to_board(component)
            
            # Progress indicator
            if iteration % 25 == 0:
                total_energy = sum(
                    math.sqrt(c.vx**2 + c.vy**2) 
                    for c in self.components.values()
                )
                overlaps = self.count_overlaps()
                print(f"  Iteration {iteration}: Energy={total_energy:.2f}, Overlaps={overlaps}")
        
        final_overlaps = self.count_overlaps()
        print(f"  Force-directed placement complete. Final overlaps: {final_overlaps}")
        
        if final_overlaps > 0:
            print(f"  WARNING: {final_overlaps} overlapping pairs remain. Manual adjustment may be needed.")
        
        return self.components


class PlacementEngine:
    """Main placement engine that orchestrates the placement process."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.groups: Dict[str, ComponentGroup] = {}
        self.components: Dict[str, Component] = {}
        self.parse_config()
    
    def load_config(self) -> dict:
        """Load YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def parse_config(self):
        """Parse the configuration into groups and components."""
        board = self.config.get('board', {})
        self.board_width = board.get('width', 50.0)
        self.board_height = board.get('height', 45.0)
        self.board_origin = tuple(board.get('origin', [0.0, 0.0]))
        
        # Parse force-directed settings
        fd_config = self.config.get('force_directed', {})
        self.fd_config = ForceDirectedConfig(
            enabled=fd_config.get('enabled', True),
            iterations=fd_config.get('iterations', 100),
            attraction_strength=fd_config.get('attraction_strength', 0.1),
            repulsion_strength=fd_config.get('repulsion_strength', 500.0),
            damping=fd_config.get('damping', 0.85),
            collision_strength=fd_config.get('collision_strength', 50.0),
            min_clearance=fd_config.get('min_clearance', 0.5),
        )
        
        # Parse component sizes from config if provided
        sizes_config = self.config.get('component_sizes', {})
        for ref, size in sizes_config.items():
            if isinstance(size, list) and len(size) == 2:
                COMPONENT_SIZES[ref] = tuple(size)
        
        # Parse groups
        groups_config = self.config.get('groups', {})
        for group_name, group_data in groups_config.items():
            group = ComponentGroup(
                name=group_name,
                anchor=group_data.get('anchor', ''),
                position=tuple(group_data.get('position', [0.0, 0.0])),
                components=group_data.get('components', {}),
            )
            self.groups[group_name] = group
            
            # Get group default layer
            group_layer = group_data.get('layer', 'front')
            
            # Create component objects
            for ref, comp_data in group.components.items():
                offset = comp_data.get('offset', [0.0, 0.0])
                size = comp_data.get('size', get_component_size(ref))
                if isinstance(size, list):
                    size = tuple(size)
                
                # Component layer overrides group layer
                comp_layer = comp_data.get('layer', group_layer)
                
                component = Component(
                    ref=ref,
                    x=group.position[0] + offset[0],
                    y=group.position[1] + offset[1],
                    rotation=comp_data.get('rotation', 0),
                    width=size[0] if isinstance(size, tuple) else size,
                    height=size[1] if isinstance(size, tuple) else size,
                    group=group_name,
                    note=comp_data.get('note', ''),
                    fixed=(ref == group.anchor),
                    layer=comp_layer,
                )
                self.components[ref] = component

        
        # Parse connections
        self.connections = self.config.get('connections', [])
    
    def check_all_overlaps(self) -> List[Tuple[str, str]]:
        """Check for all overlapping component pairs."""
        overlaps = []
        refs = list(self.components.keys())
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                c1 = self.components[refs[i]]
                c2 = self.components[refs[j]]
                if c1.overlaps(c2, margin=0):
                    overlaps.append((refs[i], refs[j]))
        return overlaps
    
    def optimize_placement(self):
        """Run force-directed optimization on all components."""
        if not self.fd_config.enabled:
            print("Force-directed optimization disabled.")
            return
        
        placer = ForceDirectedPlacer(
            self.fd_config,
            self.board_width,
            self.board_height
        )
        
        # Add ALL components, not just anchors
        for ref, comp in self.components.items():
            placer.add_component(Component(
                ref=ref,
                x=comp.x,
                y=comp.y,
                width=comp.width,
                height=comp.height,
                fixed=comp.fixed,
            ))
        
        # Add connections
        for conn in self.connections:
            if len(conn) >= 2:
                comp1, comp2 = conn[0], conn[1]
                weight = conn[2] if len(conn) > 2 else 1.0
                placer.add_connection(comp1, comp2, weight)
        
        # Run optimization
        optimized = placer.run()
        
        # Update component positions
        for ref, opt_comp in optimized.items():
            if ref in self.components:
                self.components[ref].x = opt_comp.x
                self.components[ref].y = opt_comp.y
    
    def resolve_overlaps(self, max_iterations: int = 50):
        """Post-processing to resolve any remaining overlaps."""
        print("\n=== Resolving Remaining Overlaps ===")
        
        for iteration in range(max_iterations):
            overlaps = self.check_all_overlaps()
            if not overlaps:
                print(f"  All overlaps resolved after {iteration} iterations")
                return
            
            # Push apart each overlapping pair
            for ref1, ref2 in overlaps:
                c1 = self.components[ref1]
                c2 = self.components[ref2]
                
                if c1.fixed and c2.fixed:
                    continue
                
                # Calculate push direction
                dx = c1.x - c2.x
                dy = c1.y - c2.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 0.1:
                    angle = random.uniform(0, 2 * math.pi)
                    dx, dy = math.cos(angle), math.sin(angle)
                    dist = 1.0
                
                # Calculate required separation
                min_sep_x = (c1.width + c2.width) / 2 + self.fd_config.min_clearance
                min_sep_y = (c1.height + c2.height) / 2 + self.fd_config.min_clearance
                min_sep = max(min_sep_x, min_sep_y)
                
                # Push amount
                push = (min_sep - dist) / 2 + 0.5
                
                # Apply push
                if not c1.fixed:
                    c1.x += (dx / dist) * push
                    c1.y += (dy / dist) * push
                if not c2.fixed:
                    c2.x -= (dx / dist) * push
                    c2.y -= (dy / dist) * push
        
        remaining = len(self.check_all_overlaps())
        if remaining > 0:
            print(f"  WARNING: {remaining} overlaps remain after {max_iterations} iterations")
    
    def get_all_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get final positions for all components."""
        return {
            ref: (comp.x, comp.y, comp.rotation)
            for ref, comp in self.components.items()
        }
    
    def print_overlap_report(self):
        """Print a report of any remaining overlaps."""
        overlaps = self.check_all_overlaps()
        if overlaps:
            print(f"\n⚠️  OVERLAP REPORT: {len(overlaps)} overlapping pairs")
            for ref1, ref2 in overlaps[:10]:  # Show first 10
                c1, c2 = self.components[ref1], self.components[ref2]
                dist = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
                print(f"  {ref1} <-> {ref2}: distance={dist:.1f}mm")
            if len(overlaps) > 10:
                print(f"  ... and {len(overlaps) - 10} more")
        else:
            print("\n✓ No overlapping components detected")


class KiCadPCBUpdater:
    """Updates KiCad PCB file with new component positions."""
    
    def __init__(self, pcb_path: str):
        self.pcb_path = pcb_path
        with open(pcb_path, 'r') as f:
            self.content = f.read()
    
    def update_footprint_position(self, ref_designator: str, new_x: float, new_y: float, rotation: float = None) -> bool:
        """Update the position of a footprint in the KiCad PCB content."""
        ref_pattern = rf'\(property "Reference" "{re.escape(ref_designator)}"'
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
                    new_at = f"(at {new_x} {new_y} {rotation})"
                elif at_match.group(3):
                    new_at = f"(at {new_x} {new_y}{at_match.group(3)})"
                else:
                    new_at = f"(at {new_x} {new_y})"
                
                at_absolute_start = fp_absolute_start + at_match.start()
                at_absolute_end = fp_absolute_start + at_match.end()
                self.content = self.content[:at_absolute_start] + new_at + self.content[at_absolute_end:]
                
                return True
        
        return False
    
    def save(self, output_path: str):
        """Save the modified PCB to a file."""
        with open(output_path, 'w') as f:
            f.write(self.content)
        print(f"Saved: {output_path}")


def find_latest_pcb(build_dir: str) -> Optional[str]:
    """Find the latest KiCad PCB file in the build directory."""
    import glob
    
    pattern = os.path.join(build_dir, "*.kicad_pcb")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith("_placed.kicad_pcb")]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description='KiCad PCB Component Auto-Placer with Collision Detection')
    parser.add_argument('--config', default='placement_config.yaml', 
                        help='Path to placement configuration YAML')
    parser.add_argument('--pcb', help='Path to input KiCad PCB file')
    parser.add_argument('--output', help='Path to output KiCad PCB file')
    parser.add_argument('--optimize', action='store_true',
                        help='Enable force-directed optimization')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                        help='Disable force-directed optimization')
    parser.add_argument('--resolve-overlaps', action='store_true', default=True,
                        help='Run post-processing to resolve overlaps')
    parser.set_defaults(optimize=True)
    
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_dir, args.config)
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading configuration: {config_path}")
    engine = PlacementEngine(config_path)
    
    engine.fd_config.enabled = args.optimize
    
    # Initial overlap check
    initial_overlaps = engine.check_all_overlaps()
    print(f"\nInitial configuration: {len(initial_overlaps)} overlapping pairs")
    
    if args.optimize:
        print("\n=== Force-Directed Optimization ===")
        engine.optimize_placement()
    
    if args.resolve_overlaps:
        engine.resolve_overlaps()
    
    # Final overlap report
    engine.print_overlap_report()
    
    # Get final positions
    positions = engine.get_all_positions()
    
    # Find PCB file
    if args.pcb:
        pcb_path = args.pcb
    else:
        build_dir = os.path.join(project_dir, "build/builds/default")
        pcb_path = find_latest_pcb(build_dir)
    
    if not pcb_path or not os.path.exists(pcb_path):
        print(f"Error: PCB file not found")
        print(f"  Searched: {build_dir}")
        sys.exit(1)
    
    print(f"\nUpdating PCB: {pcb_path}")
    updater = KiCadPCBUpdater(pcb_path)
    
    updated = 0
    not_found = []
    
    for ref, (x, y, rotation) in positions.items():
        if updater.update_footprint_position(ref, x, y, rotation):
            print(f"  ✓ {ref}: ({x:.1f}, {y:.1f}, {rotation}°)")
            updated += 1
        else:
            not_found.append(ref)
    
    if args.output:
        output_path = args.output
    else:
        output_path = pcb_path.replace('.kicad_pcb', '_placed.kicad_pcb')
    
    updater.save(output_path)
    
    print(f"\n{'='*50}")
    print(f"Placement Summary")
    print(f"{'='*50}")
    print(f"Groups: {len(engine.groups)}")
    print(f"Components updated: {updated}")
    print(f"Components not found: {len(not_found)}")
    if not_found:
        print(f"  Missing: {', '.join(not_found)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
