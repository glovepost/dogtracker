#!/usr/bin/env python3
"""
Advanced KiCad PCB Component Auto-Placer with Design Block Templates

Features:
- Reusable design block templates (IC + passives grouped)
- YAML configuration for easy editing
- Force-directed algorithm for block placement optimization
- Bounding box collision detection with layer awareness
- Block-level and component-level collision resolution

Usage:
    python autoplace_blocks.py [--config block_templates.yaml] [--optimize]
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
# DATA STRUCTURES
# ============================================================================

@dataclass
class Component:
    """Represents a single component with position, size, and properties."""
    ref: str
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    width: float = 2.0
    height: float = 2.0
    layer: str = "front"
    note: str = ""
    block_name: Optional[str] = None  # Which block this belongs to
    
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
        if self.layer != other.layer:
            return False
        return (
            self.left - margin < other.right + margin and
            self.right + margin > other.left - margin and
            self.bottom - margin < other.top + margin and
            self.top + margin > other.bottom - margin
        )


@dataclass
class DesignBlock:
    """A reusable design block containing IC + associated passives."""
    name: str
    anchor: str
    width: float
    height: float
    layer: str = "front"
    components: Dict[str, dict] = field(default_factory=dict)
    
    # Instance position (set when instantiated)
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0  # Velocity for force-directed
    vy: float = 0.0
    fixed: bool = False
    
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
    
    def overlaps(self, other: 'DesignBlock', margin: float = 1.0) -> bool:
        """Check if this block overlaps with another."""
        if self.layer != other.layer:
            return False
        return (
            self.left - margin < other.right + margin and
            self.right + margin > other.left - margin and
            self.bottom - margin < other.top + margin and
            self.top + margin > other.bottom - margin
        )
    
    def get_component_positions(self) -> Dict[str, Tuple[float, float, float, str]]:
        """Get absolute positions for all components in this block.
        Returns: {ref: (x, y, rotation, layer)}"""
        positions = {}
        for ref, comp_data in self.components.items():
            offset = comp_data.get('offset', [0.0, 0.0])
            rotation = comp_data.get('rotation', 0)
            comp_layer = comp_data.get('layer', self.layer)
            positions[ref] = (
                self.x + offset[0],
                self.y + offset[1],
                rotation,
                comp_layer
            )
        return positions


@dataclass
class ForceDirectedConfig:
    """Configuration for force-directed placement algorithm."""
    enabled: bool = True
    iterations: int = 150
    attraction_strength: float = 0.2
    repulsion_strength: float = 400.0
    damping: float = 0.78
    collision_strength: float = 50.0
    min_clearance: float = 0.5


# ============================================================================
# FORCE-DIRECTED BLOCK PLACER
# ============================================================================

class BlockPlacer:
    """Force-directed placement algorithm for design blocks."""
    
    def __init__(self, config: ForceDirectedConfig, board_width: float, board_height: float):
        self.config = config
        self.board_width = board_width
        self.board_height = board_height
        self.blocks: Dict[str, DesignBlock] = {}
    
    def add_block(self, block: DesignBlock):
        """Add a design block to the placer."""
        self.blocks[block.name] = block
    
    def calculate_collision_force(self, b1: DesignBlock, b2: DesignBlock) -> Tuple[float, float]:
        """Calculate force to push apart overlapping blocks."""
        if not b1.overlaps(b2, margin=self.config.min_clearance):
            return 0.0, 0.0
        
        dx = b1.x - b2.x
        dy = b1.y - b2.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 0.1:
            angle = random.uniform(0, 2 * math.pi)
            dx, dy = math.cos(angle), math.sin(angle)
            distance = 1.0
        
        # Calculate overlap amount
        overlap_x = min(b1.right, b2.right) - max(b1.left, b2.left)
        overlap_y = min(b1.top, b2.top) - max(b1.bottom, b2.bottom)
        overlap = max(overlap_x, overlap_y) + self.config.min_clearance
        
        force = self.config.collision_strength * overlap
        fx = (dx / distance) * force
        fy = (dy / distance) * force
        
        return fx, fy
    
    def calculate_repulsion_force(self, b1: DesignBlock, b2: DesignBlock) -> Tuple[float, float]:
        """Calculate repulsion force between blocks."""
        dx = b1.x - b2.x
        dy = b1.y - b2.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        min_dist = (b1.width + b2.width) / 2 + self.config.min_clearance
        
        if distance < 0.1:
            distance = 0.1
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)
        
        if distance < min_dist:
            force = self.config.repulsion_strength * (min_dist / distance)**2
        else:
            force = self.config.repulsion_strength / (distance * distance)
        
        fx = (dx / distance) * force
        fy = (dy / distance) * force
        
        return fx, fy
    
    def constrain_to_board(self, block: DesignBlock):
        """Keep block within board boundaries."""
        margin_x = block.width / 2 + 1.0
        margin_y = block.height / 2 + 1.0
        
        half_w = self.board_width / 2 - margin_x
        half_h = self.board_height / 2 - margin_y
        
        block.x = max(-half_w, min(half_w, block.x))
        block.y = max(-half_h, min(half_h, block.y))
    
    def count_overlaps(self) -> int:
        """Count overlapping block pairs."""
        overlaps = 0
        names = list(self.blocks.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                b1 = self.blocks[names[i]]
                b2 = self.blocks[names[j]]
                if b1.overlaps(b2, margin=0):
                    overlaps += 1
        return overlaps
    
    def run(self) -> Dict[str, DesignBlock]:
        """Run force-directed placement on blocks."""
        if not self.config.enabled:
            return self.blocks
        
        print(f"Running block placement ({self.config.iterations} iterations)...")
        print(f"  Initial overlaps: {self.count_overlaps()}")
        
        for iteration in range(self.config.iterations):
            forces: Dict[str, Tuple[float, float]] = {name: (0.0, 0.0) for name in self.blocks}
            
            names = list(self.blocks.keys())
            
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    b1 = self.blocks[names[i]]
                    b2 = self.blocks[names[j]]
                    
                    # Skip if on different layers
                    if b1.layer != b2.layer:
                        continue
                    
                    # Collision forces
                    cfx, cfy = self.calculate_collision_force(b1, b2)
                    if cfx != 0 or cfy != 0:
                        if not b1.fixed:
                            forces[names[i]] = (forces[names[i]][0] + cfx, forces[names[i]][1] + cfy)
                        if not b2.fixed:
                            forces[names[j]] = (forces[names[j]][0] - cfx, forces[names[j]][1] - cfy)
                    else:
                        # Repulsion forces
                        fx, fy = self.calculate_repulsion_force(b1, b2)
                        if not b1.fixed:
                            forces[names[i]] = (forces[names[i]][0] + fx, forces[names[i]][1] + fy)
                        if not b2.fixed:
                            forces[names[j]] = (forces[names[j]][0] - fx, forces[names[j]][1] - fy)
            
            # Apply forces
            for name, (fx, fy) in forces.items():
                block = self.blocks[name]
                if block.fixed:
                    continue
                
                block.vx = (block.vx + fx) * self.config.damping
                block.vy = (block.vy + fy) * self.config.damping
                
                max_velocity = 3.0
                speed = math.sqrt(block.vx**2 + block.vy**2)
                if speed > max_velocity:
                    block.vx = (block.vx / speed) * max_velocity
                    block.vy = (block.vy / speed) * max_velocity
                
                block.x += block.vx
                block.y += block.vy
                
                self.constrain_to_board(block)
            
            if iteration % 30 == 0:
                overlaps = self.count_overlaps()
                print(f"  Iteration {iteration}: Overlaps={overlaps}")
        
        print(f"  Block placement complete. Final overlaps: {self.count_overlaps()}")
        return self.blocks


# ============================================================================
# DESIGN BLOCK ENGINE
# ============================================================================

class DesignBlockEngine:
    """Main engine for block-based component placement."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.block_templates: Dict[str, dict] = {}
        self.block_instances: List[DesignBlock] = []
        self.standalone_components: Dict[str, Component] = {}
        self.all_components: Dict[str, Component] = {}
        self.parse_config()
    
    def load_config(self) -> dict:
        """Load YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def parse_config(self):
        """Parse the configuration into blocks and components."""
        # Board settings
        board = self.config.get('board', {})
        self.board_width = board.get('width', 40.0)
        self.board_height = board.get('height', 35.0)
        
        # Force-directed settings
        fd_config = self.config.get('force_directed', {})
        self.fd_config = ForceDirectedConfig(
            enabled=fd_config.get('enabled', True),
            iterations=fd_config.get('iterations', 150),
            attraction_strength=fd_config.get('attraction_strength', 0.2),
            repulsion_strength=fd_config.get('repulsion_strength', 400.0),
            damping=fd_config.get('damping', 0.78),
            collision_strength=fd_config.get('collision_strength', 50.0),
            min_clearance=fd_config.get('min_clearance', 0.5),
        )
        
        # Parse block templates
        self.block_templates = self.config.get('blocks', {})
        
        # Parse block instances
        instances = self.config.get('instances', [])
        for instance in instances:
            block_name = instance.get('block')
            if block_name not in self.block_templates:
                print(f"Warning: Block template '{block_name}' not found")
                continue
            
            template = self.block_templates[block_name]
            position = instance.get('position', [0.0, 0.0])
            layer = instance.get('layer', template.get('layer', 'front'))
            
            size = template.get('size', [10.0, 10.0])
            
            block = DesignBlock(
                name=f"{block_name}_{len(self.block_instances)}",
                anchor=template.get('anchor', ''),
                width=size[0],
                height=size[1],
                layer=layer,
                components=template.get('components', {}),
                x=position[0],
                y=position[1],
            )
            self.block_instances.append(block)
        
        # Parse standalone components
        standalone = self.config.get('standalone', {})
        for ref, comp_data in standalone.items():
            position = comp_data.get('position', [0.0, 0.0])
            size = comp_data.get('size', [2.0, 2.0])
            
            component = Component(
                ref=ref,
                x=position[0],
                y=position[1],
                width=size[0] if isinstance(size, list) else size,
                height=size[1] if isinstance(size, list) else size,
                layer=comp_data.get('layer', 'front'),
            )
            self.standalone_components[ref] = component
    
    def optimize_block_placement(self):
        """Run force-directed optimization on blocks."""
        if not self.fd_config.enabled:
            print("Block optimization disabled.")
            return
        
        placer = BlockPlacer(
            self.fd_config,
            self.board_width,
            self.board_height
        )
        
        for block in self.block_instances:
            placer.add_block(block)
        
        optimized = placer.run()
        
        # Update block positions
        for i, block in enumerate(self.block_instances):
            if block.name in optimized:
                opt_block = optimized[block.name]
                block.x = opt_block.x
                block.y = opt_block.y
    
    def resolve_overlaps(self, max_iterations: int = 50):
        """Resolve any remaining block overlaps."""
        print("\n=== Resolving Block Overlaps ===")
        
        for iteration in range(max_iterations):
            overlaps = []
            for i in range(len(self.block_instances)):
                for j in range(i + 1, len(self.block_instances)):
                    b1 = self.block_instances[i]
                    b2 = self.block_instances[j]
                    if b1.overlaps(b2, margin=0):
                        overlaps.append((i, j))
            
            if not overlaps:
                print(f"  All block overlaps resolved after {iteration} iterations")
                return
            
            for i, j in overlaps:
                b1 = self.block_instances[i]
                b2 = self.block_instances[j]
                
                dx = b1.x - b2.x
                dy = b1.y - b2.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 0.1:
                    angle = random.uniform(0, 2 * math.pi)
                    dx, dy = math.cos(angle), math.sin(angle)
                    dist = 1.0
                
                min_sep = (b1.width + b2.width) / 2 + self.fd_config.min_clearance
                push = (min_sep - dist) / 2 + 0.5
                
                b1.x += (dx / dist) * push
                b1.y += (dy / dist) * push
                b2.x -= (dx / dist) * push
                b2.y -= (dy / dist) * push
        
        remaining = sum(1 for i in range(len(self.block_instances)) 
                       for j in range(i+1, len(self.block_instances)) 
                       if self.block_instances[i].overlaps(self.block_instances[j], margin=0))
        if remaining > 0:
            print(f"  WARNING: {remaining} block overlaps remain")
    
    def get_all_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get final positions for all components."""
        positions = {}
        
        # Get component positions from blocks
        for block in self.block_instances:
            block_positions = block.get_component_positions()
            for ref, (x, y, rotation, layer) in block_positions.items():
                positions[ref] = (x, y, rotation)
        
        # Get standalone component positions
        for ref, comp in self.standalone_components.items():
            positions[ref] = (comp.x, comp.y, comp.rotation)
        
        return positions
    
    def print_block_summary(self):
        """Print summary of block placements."""
        print("\n" + "="*50)
        print("Design Block Summary")
        print("="*50)
        for block in self.block_instances:
            print(f"  {block.name}:")
            print(f"    Position: ({block.x:.1f}, {block.y:.1f})")
            print(f"    Size: {block.width}x{block.height}mm")
            print(f"    Layer: {block.layer}")
            print(f"    Components: {len(block.components)}")


# ============================================================================
# KICAD PCB UPDATER
# ============================================================================

class KiCadPCBUpdater:
    """Updates KiCad PCB file with new component positions."""
    
    def __init__(self, pcb_path: str):
        self.pcb_path = pcb_path
        with open(pcb_path, 'r') as f:
            self.content = f.read()
    
    def update_footprint_position(self, ref_designator: str, new_x: float, new_y: float, 
                                   rotation: float = None) -> bool:
        """Update the position of a footprint."""
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


# ============================================================================
# MAIN
# ============================================================================

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
    parser = argparse.ArgumentParser(description='KiCad PCB Block-Based Auto-Placer')
    parser.add_argument('--config', default='block_templates.yaml', 
                        help='Path to block templates YAML')
    parser.add_argument('--pcb', help='Path to input KiCad PCB file')
    parser.add_argument('--output', help='Path to output KiCad PCB file')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='Enable force-directed optimization')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                        help='Disable force-directed optimization')
    
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_dir, args.config)
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading block templates: {config_path}")
    engine = DesignBlockEngine(config_path)
    
    engine.fd_config.enabled = args.optimize
    
    print(f"\nLoaded {len(engine.block_instances)} design blocks:")
    for block in engine.block_instances:
        print(f"  - {block.name}: {len(block.components)} components")
    print(f"Standalone components: {len(engine.standalone_components)}")
    
    if args.optimize:
        print("\n=== Block Placement Optimization ===")
        engine.optimize_block_placement()
    
    engine.resolve_overlaps()
    engine.print_block_summary()
    
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
    print(f"Design blocks: {len(engine.block_instances)}")
    print(f"Total components: {len(positions)}")
    print(f"Components updated: {updated}")
    print(f"Components not found: {len(not_found)}")
    if not_found:
        print(f"  Missing: {', '.join(not_found)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
