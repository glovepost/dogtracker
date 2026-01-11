#!/usr/bin/env python3
"""
Module-Based PCB Placer for Atopile Sync Groups

This script implements the module-based layout workflow for atopile projects:
1. Reads module definitions from ato.yaml
2. Builds each module individually using atopile CLI
3. Creates layout directories for each module
4. Places components in each module's PCB file
5. Enables "Pull Group" workflow in KiCad

Usage:
    python module_placer.py [--modules all|module1,module2] [--build] [--place]

The workflow:
    1. Run this script to build modules and create layouts
    2. Open main PCB in KiCad
    3. Select components belonging to a module
    4. Click "Pull Group" to import the module's layout
"""

import os
import sys
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import glob as glob_module

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
    logger.info("Using KiCad pcbnew API")
except ImportError:
    logger.warning("pcbnew not available - limited functionality")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModuleDefinition:
    """Represents a module from ato.yaml"""
    name: str
    entry: str
    components: List[str] = field(default_factory=list)
    layout_dir: str = ""
    pcb_path: str = ""


@dataclass
class ComponentPlacement:
    """Component placement info from config"""
    ref: str
    x: float
    y: float
    rotation: float = 0.0
    width: float = 2.0
    height: float = 2.0
    layer: str = "front"


# ============================================================================
# ATO.YAML PARSER
# ============================================================================

class AtoYamlParser:
    """Parses ato.yaml to extract module definitions"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.ato_yaml_path = os.path.join(project_dir, "ato.yaml")
        self.modules: Dict[str, ModuleDefinition] = {}

    def parse(self) -> Dict[str, ModuleDefinition]:
        """Parse ato.yaml and extract module builds"""
        if not os.path.exists(self.ato_yaml_path):
            raise FileNotFoundError(f"ato.yaml not found: {self.ato_yaml_path}")

        with open(self.ato_yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        builds = config.get('builds', {})

        for build_name, build_config in builds.items():
            # Skip the default build - it's the main project
            if build_name == 'default':
                continue

            entry = build_config.get('entry', '')
            if not entry:
                continue

            module = ModuleDefinition(
                name=build_name,
                entry=entry,
                layout_dir=os.path.join(self.project_dir, "layouts", build_name),
                pcb_path=os.path.join(self.project_dir, "layouts", build_name, f"{build_name}.kicad_pcb")
            )
            self.modules[build_name] = module
            logger.debug(f"Found module: {build_name} -> {entry}")

        logger.info(f"Parsed {len(self.modules)} module builds from ato.yaml")
        return self.modules


# ============================================================================
# PLACEMENT CONFIG PARSER
# ============================================================================

class PlacementConfigParser:
    """Parses placement_config.yaml to get module-component mappings"""

    # Map group names to module names from ato.yaml
    GROUP_TO_MODULE = {
        'power_filter': 'power_filter',
        'i2c_bus': 'i2c_bus',
        'accelerometer': 'lis3dhtr',
        'imu': 'qmi8658c',
        'humidity': 'hdc2080',
        'eeprom': 'eeprom',
        'rf_matching': 'rf_matching',
        'status_led': 'status_led',
        'ntc_temp': 'ntc_temp',
        'swd_debug': 'swd_debug',
    }

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.groups: Dict[str, Dict] = {}
        self.board_config: Dict = {}

    def parse(self) -> Dict[str, List[ComponentPlacement]]:
        """Parse placement config and return module -> components mapping"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Placement config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.board_config = config.get('board', {})
        self.groups = config.get('groups', {})

        module_components: Dict[str, List[ComponentPlacement]] = {}

        for group_name, group_data in self.groups.items():
            # Map group name to module name
            module_name = self.GROUP_TO_MODULE.get(group_name)
            if not module_name:
                logger.debug(f"Skipping group '{group_name}' - no module mapping")
                continue

            components = []
            group_pos = group_data.get('position', [0.0, 0.0])
            group_layer = group_data.get('layer', 'front')

            for ref, comp_data in group_data.get('components', {}).items():
                offset = comp_data.get('offset', [0.0, 0.0])
                size = comp_data.get('size', [2.0, 2.0])

                placement = ComponentPlacement(
                    ref=ref,
                    x=offset[0],  # Relative to module origin
                    y=offset[1],
                    rotation=comp_data.get('rotation', 0),
                    width=size[0] if isinstance(size, list) else size,
                    height=size[1] if isinstance(size, list) else size,
                    layer=comp_data.get('layer', group_layer)
                )
                components.append(placement)

            module_components[module_name] = components
            logger.debug(f"Module '{module_name}': {len(components)} components")

        logger.info(f"Parsed placements for {len(module_components)} modules")
        return module_components


# ============================================================================
# ATOPILE BUILD MANAGER
# ============================================================================

class AtopileBuildManager:
    """Manages atopile builds for modules"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir

    def build_module(self, module_name: str) -> Tuple[bool, str]:
        """Build a single module using atopile CLI

        Returns:
            Tuple of (success, output_pcb_path or error_message)
        """
        logger.info(f"Building module: {module_name}")

        try:
            # Run ato build for this specific target
            result = subprocess.run(
                ['ato', 'build', '--target', module_name],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                logger.error(f"Build failed for {module_name}")
                logger.error(result.stderr)
                return False, result.stderr

            logger.info(f"  Build successful: {module_name}")

            # Find the generated PCB file
            build_dir = os.path.join(self.project_dir, "build", "builds", module_name)
            pcb_files = glob_module.glob(os.path.join(build_dir, "*.kicad_pcb"))

            if pcb_files:
                # Get the most recent one (excluding _placed versions)
                pcb_files = [f for f in pcb_files if '_placed' not in f]
                if pcb_files:
                    pcb_path = max(pcb_files, key=os.path.getmtime)
                    return True, pcb_path

            return True, ""

        except subprocess.TimeoutExpired:
            return False, "Build timed out"
        except Exception as e:
            return False, str(e)

    def build_all_modules(self, modules: Dict[str, ModuleDefinition]) -> Dict[str, str]:
        """Build all modules and return mapping of module_name -> pcb_path"""
        results = {}

        for module_name in modules:
            success, result = self.build_module(module_name)
            if success:
                results[module_name] = result
            else:
                logger.warning(f"Failed to build {module_name}: {result}")

        return results


# ============================================================================
# LAYOUT DIRECTORY MANAGER
# ============================================================================

class LayoutManager:
    """Manages layout directories for modules"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.layouts_dir = os.path.join(project_dir, "layouts")

    def create_module_layout_dir(self, module_name: str) -> str:
        """Create layout directory for a module"""
        layout_dir = os.path.join(self.layouts_dir, module_name)
        os.makedirs(layout_dir, exist_ok=True)

        # Create fp-lib-table if it doesn't exist
        fp_lib_table = os.path.join(layout_dir, "fp-lib-table")
        if not os.path.exists(fp_lib_table):
            # Copy from default layout or create minimal one
            default_fp_lib = os.path.join(self.layouts_dir, "default", "fp-lib-table")
            if os.path.exists(default_fp_lib):
                shutil.copy(default_fp_lib, fp_lib_table)
            else:
                with open(fp_lib_table, 'w') as f:
                    f.write('(fp_lib_table\n  (version 7)\n)\n')

        logger.debug(f"Created layout directory: {layout_dir}")
        return layout_dir

    def copy_pcb_to_layout(self, build_pcb_path: str, module_name: str) -> str:
        """Copy built PCB to layout directory"""
        layout_dir = self.create_module_layout_dir(module_name)
        dest_pcb = os.path.join(layout_dir, f"{module_name}.kicad_pcb")

        if os.path.exists(build_pcb_path):
            shutil.copy(build_pcb_path, dest_pcb)
            logger.info(f"  Copied PCB to: {dest_pcb}")
            return dest_pcb
        else:
            logger.warning(f"  Source PCB not found: {build_pcb_path}")
            return ""


# ============================================================================
# MODULE PCB PLACER
# ============================================================================

class ModulePCBPlacer:
    """Places components within a module's PCB file using auto-arrangement"""

    def __init__(self, pcb_path: str):
        self.pcb_path = pcb_path
        self.board = None
        self.components: Dict[str, dict] = {}

        if PCBNEW_AVAILABLE and os.path.exists(pcb_path):
            try:
                self.board = pcbnew.LoadBoard(pcb_path)
                self._extract_components()
                logger.debug(f"Loaded PCB: {pcb_path}")
            except Exception as e:
                logger.warning(f"Failed to load PCB: {e}")

    def _extract_components(self):
        """Extract components from PCB with their properties"""
        if not self.board:
            return

        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            bbox = fp.GetFpPadsLocalBbox()
            width = max(pcbnew.ToMM(bbox.GetWidth()), 1.0)
            height = max(pcbnew.ToMM(bbox.GetHeight()), 1.0)

            # Add clearance
            width += 0.5
            height += 0.5

            self.components[ref] = {
                'width': width,
                'height': height,
                'x': pcbnew.ToMM(fp.GetPosition().x),
                'y': pcbnew.ToMM(fp.GetPosition().y),
                'rotation': fp.GetOrientationDegrees(),
            }

        logger.debug(f"  Extracted {len(self.components)} components")

    def auto_arrange(self, center: Tuple[float, float] = (100.0, 100.0),
                     spacing: float = 2.0) -> int:
        """Auto-arrange components in a compact grid layout

        Args:
            center: Center point for the arrangement (mm)
            spacing: Minimum spacing between components (mm)

        Returns:
            Number of components arranged
        """
        if not self.board or not self.components:
            return 0

        # Sort components: ICs first, then by ref
        sorted_refs = sorted(self.components.keys(),
                           key=lambda r: (0 if r.startswith('U') else 1, r))

        # Simple grid-based placement
        # Find the main IC (usually U1) and place it at center
        arranged = 0
        current_x = center[0]
        current_y = center[1]
        row_height = 0
        row_start_x = center[0]

        for i, ref in enumerate(sorted_refs):
            comp = self.components[ref]

            try:
                fp = self.board.FindFootprintByReference(ref)
                if fp is None:
                    continue

                if ref.startswith('U'):
                    # Place IC at center
                    pos = pcbnew.VECTOR2I(
                        int(pcbnew.FromMM(center[0])),
                        int(pcbnew.FromMM(center[1]))
                    )
                    # Reset row position after IC
                    current_x = center[0] - 10
                    current_y = center[1] + comp['height'] / 2 + spacing + 2
                    row_start_x = current_x
                else:
                    # Place passive components in rows below/around the IC
                    if current_x > center[0] + 15:
                        # New row
                        current_x = row_start_x
                        current_y += row_height + spacing
                        row_height = 0

                    pos = pcbnew.VECTOR2I(
                        int(pcbnew.FromMM(current_x + comp['width'] / 2)),
                        int(pcbnew.FromMM(current_y))
                    )

                    current_x += comp['width'] + spacing
                    row_height = max(row_height, comp['height'])

                fp.SetPosition(pos)
                arranged += 1
                logger.debug(f"  Arranged {ref} at ({pcbnew.ToMM(pos.x):.1f}, {pcbnew.ToMM(pos.y):.1f})")

            except Exception as e:
                logger.warning(f"  Failed to arrange {ref}: {e}")

        return arranged

    def place_components(self, placements: List[ComponentPlacement],
                         center_offset: Tuple[float, float] = (50.0, 50.0)) -> int:
        """Place components in the PCB using explicit placements

        Args:
            placements: List of component placements (relative to module origin)
            center_offset: Offset from PCB origin to module center (mm)

        Returns:
            Number of components successfully placed
        """
        if not self.board:
            logger.warning("No board loaded - cannot place components")
            return 0

        placed = 0

        for placement in placements:
            try:
                fp = self.board.FindFootprintByReference(placement.ref)
                if fp is None:
                    logger.debug(f"  Footprint not found: {placement.ref}")
                    continue

                # Calculate absolute position (center_offset + relative offset)
                abs_x = center_offset[0] + placement.x
                abs_y = center_offset[1] + placement.y

                # Convert mm to KiCad internal units
                pos = pcbnew.VECTOR2I(
                    int(pcbnew.FromMM(abs_x)),
                    int(pcbnew.FromMM(abs_y))
                )
                fp.SetPosition(pos)

                # Set rotation
                fp.SetOrientationDegrees(placement.rotation)

                # Set layer if back
                if placement.layer.lower() == "back":
                    fp.SetLayerAndFlip(pcbnew.B_Cu)

                logger.debug(f"  Placed {placement.ref} at ({abs_x:.1f}, {abs_y:.1f})")
                placed += 1

            except Exception as e:
                logger.warning(f"  Failed to place {placement.ref}: {e}")

        return placed

    def save(self, output_path: str = None):
        """Save the PCB"""
        if not self.board:
            return

        save_path = output_path or self.pcb_path
        try:
            self.board.Save(save_path)
            logger.info(f"  Saved PCB: {save_path}")
        except Exception as e:
            logger.error(f"  Failed to save: {e}")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class ModulePlacerOrchestrator:
    """Orchestrates the complete module placement workflow"""

    def __init__(self, project_dir: str, placement_config: str = "placement_config.yaml"):
        self.project_dir = project_dir
        self.placement_config_path = os.path.join(project_dir, placement_config)

        # Initialize parsers
        self.ato_parser = AtoYamlParser(project_dir)
        self.placement_parser = PlacementConfigParser(self.placement_config_path)
        self.build_manager = AtopileBuildManager(project_dir)
        self.layout_manager = LayoutManager(project_dir)

        # Parsed data
        self.modules: Dict[str, ModuleDefinition] = {}
        self.module_placements: Dict[str, List[ComponentPlacement]] = {}

    def parse_configs(self):
        """Parse all configuration files"""
        logger.info("=== Parsing Configurations ===")
        self.modules = self.ato_parser.parse()
        self.module_placements = self.placement_parser.parse()

        # Report module-placement alignment
        modules_with_placements = set(self.modules.keys()) & set(self.module_placements.keys())
        modules_without_placements = set(self.modules.keys()) - set(self.module_placements.keys())

        logger.info(f"Modules with placements: {len(modules_with_placements)}")
        if modules_without_placements:
            logger.warning(f"Modules without placements: {', '.join(modules_without_placements)}")

    def build_modules(self, module_names: Optional[List[str]] = None) -> Dict[str, str]:
        """Build specified modules (or all)

        Returns:
            Dict mapping module_name -> built_pcb_path
        """
        logger.info("\n=== Building Modules ===")

        if module_names is None or 'all' in module_names:
            modules_to_build = self.modules
        else:
            modules_to_build = {k: v for k, v in self.modules.items() if k in module_names}

        return self.build_manager.build_all_modules(modules_to_build)

    def create_layouts(self, built_pcbs: Dict[str, str]):
        """Create layout directories and copy PCBs"""
        logger.info("\n=== Creating Layout Directories ===")

        for module_name, build_pcb in built_pcbs.items():
            if build_pcb:
                self.layout_manager.copy_pcb_to_layout(build_pcb, module_name)
            else:
                # Create empty layout dir anyway
                self.layout_manager.create_module_layout_dir(module_name)

    def place_modules(self, module_names: Optional[List[str]] = None):
        """Auto-arrange components in module PCBs"""
        logger.info("\n=== Auto-Arranging Components in Module PCBs ===")

        if module_names is None or 'all' in module_names:
            modules_to_place = set(self.modules.keys())
        else:
            modules_to_place = set(module_names) & set(self.modules.keys())

        for module_name in sorted(modules_to_place):
            pcb_path = os.path.join(
                self.project_dir, "layouts", module_name, f"{module_name}.kicad_pcb"
            )

            if not os.path.exists(pcb_path):
                logger.warning(f"PCB not found for {module_name}: {pcb_path}")
                continue

            logger.info(f"Auto-arranging components in: {module_name}")

            placer = ModulePCBPlacer(pcb_path)
            arranged = placer.auto_arrange(center=(100.0, 100.0), spacing=1.5)
            placer.save()

            logger.info(f"  Arranged {arranged} components")

    def run(self, module_names: Optional[List[str]] = None,
            build: bool = True, place: bool = True):
        """Run the complete workflow

        Args:
            module_names: List of module names to process, or None for all
            build: Whether to build modules with atopile
            place: Whether to place components in PCBs
        """
        self.parse_configs()

        if build:
            built_pcbs = self.build_modules(module_names)
            self.create_layouts(built_pcbs)

        if place:
            self.place_modules(module_names)

        self.print_summary()

    def print_summary(self):
        """Print summary of results"""
        print(f"\n{'='*60}")
        print("Module Placer Summary")
        print(f"{'='*60}")
        print(f"Project: {self.project_dir}")
        print(f"Modules defined: {len(self.modules)}")
        print(f"Modules with placements: {len(self.module_placements)}")
        print()
        print("Module Layouts:")
        for module_name in sorted(self.modules.keys()):
            layout_dir = os.path.join(self.project_dir, "layouts", module_name)
            pcb_exists = os.path.exists(os.path.join(layout_dir, f"{module_name}.kicad_pcb"))
            status = "OK" if pcb_exists else "MISSING"
            placements = len(self.module_placements.get(module_name, []))
            print(f"  {module_name:20} [{status:7}] {placements:2} components")
        print()
        print("Next steps:")
        print("  1. Open the main PCB in KiCad (layouts/default/default.kicad_pcb)")
        print("  2. Select components belonging to a module")
        print("  3. Click 'Pull Group' in the atopile toolbar")
        print("  4. The module's layout will be applied")
        print(f"{'='*60}")


# ============================================================================
# SYNC TO MAIN BOARD
# ============================================================================

class MainBoardSyncer:
    """Syncs module placements to the main board"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.main_pcb = os.path.join(project_dir, "layouts", "default", "default.kicad_pcb")

    def sync_module(self, module_name: str,
                    module_placements: List[ComponentPlacement],
                    module_position: Tuple[float, float]) -> int:
        """Sync a module's placements to the main board

        Args:
            module_name: Name of the module
            module_placements: Component placements relative to module origin
            module_position: Where to place the module on the main board

        Returns:
            Number of components synced
        """
        if not os.path.exists(self.main_pcb):
            logger.error(f"Main PCB not found: {self.main_pcb}")
            return 0

        if not PCBNEW_AVAILABLE:
            logger.error("pcbnew required for main board sync")
            return 0

        try:
            board = pcbnew.LoadBoard(self.main_pcb)
        except Exception as e:
            logger.error(f"Failed to load main board: {e}")
            return 0

        synced = 0
        board_center_x = 100.0  # Typical board center
        board_center_y = 75.0

        for placement in module_placements:
            try:
                fp = board.FindFootprintByReference(placement.ref)
                if fp is None:
                    continue

                # Calculate absolute position on main board
                abs_x = board_center_x + module_position[0] + placement.x
                abs_y = board_center_y + module_position[1] + placement.y

                pos = pcbnew.VECTOR2I(
                    int(pcbnew.FromMM(abs_x)),
                    int(pcbnew.FromMM(abs_y))
                )
                fp.SetPosition(pos)
                fp.SetOrientationDegrees(placement.rotation)

                if placement.layer.lower() == "back":
                    fp.SetLayerAndFlip(pcbnew.B_Cu)

                synced += 1

            except Exception as e:
                logger.warning(f"Failed to sync {placement.ref}: {e}")

        try:
            board.Save(self.main_pcb)
            logger.info(f"Synced {synced} components to main board")
        except Exception as e:
            logger.error(f"Failed to save main board: {e}")

        return synced


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Module-Based PCB Placer for Atopile Sync Groups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build and place all modules
  python module_placer.py

  # Build and place specific modules
  python module_placer.py --modules power_filter,i2c_bus

  # Only place (skip building)
  python module_placer.py --no-build

  # Only build (skip placing)
  python module_placer.py --no-place

  # Sync all modules to main board
  python module_placer.py --sync-main
"""
    )
    parser.add_argument('--modules', default='all',
                        help='Comma-separated list of modules to process, or "all"')
    parser.add_argument('--config', default='placement_config.yaml',
                        help='Path to placement configuration')
    parser.add_argument('--build', dest='build', action='store_true', default=True,
                        help='Build modules with atopile (default)')
    parser.add_argument('--no-build', dest='build', action='store_false',
                        help='Skip building modules')
    parser.add_argument('--place', dest='place', action='store_true', default=True,
                        help='Place components in module PCBs (default)')
    parser.add_argument('--no-place', dest='place', action='store_false',
                        help='Skip placing components')
    parser.add_argument('--sync-main', action='store_true',
                        help='Sync all module placements to main board')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse module names
    if args.modules == 'all':
        module_names = None
    else:
        module_names = [m.strip() for m in args.modules.split(',')]

    # Run orchestrator
    orchestrator = ModulePlacerOrchestrator(project_dir, args.config)
    orchestrator.run(module_names, build=args.build, place=args.place)

    # Optionally sync to main board
    if args.sync_main:
        logger.info("\n=== Syncing to Main Board ===")
        syncer = MainBoardSyncer(project_dir)

        # Load placement config for module positions
        placement_parser = PlacementConfigParser(
            os.path.join(project_dir, args.config)
        )

        with open(os.path.join(project_dir, args.config), 'r') as f:
            config = yaml.safe_load(f)

        groups = config.get('groups', {})
        module_placements = placement_parser.parse()

        for group_name, group_data in groups.items():
            module_name = PlacementConfigParser.GROUP_TO_MODULE.get(group_name)
            if not module_name or module_name not in module_placements:
                continue

            position = tuple(group_data.get('position', [0.0, 0.0]))
            placements = module_placements[module_name]

            synced = syncer.sync_module(module_name, placements, position)
            logger.info(f"  {module_name}: {synced} components synced")


if __name__ == "__main__":
    main()
