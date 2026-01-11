#!/usr/bin/env python3
"""
Module Autorouter - Freerouting Integration for Module PCBs

This script autoroutes module PCB layouts using Freerouting:
1. Exports DSN files from module PCBs
2. Runs Freerouting to autoroute
3. Imports SES files back into PCBs

Prerequisites:
    - Java Runtime Environment (JRE) installed
    - Freerouting JAR (downloaded automatically if missing)
    - Run with KiCad's Python:
      /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3

Usage:
    python module_autorouter.py [--modules all|module1,module2] [--passes N] [--timeout N]
"""

import os
import sys
import subprocess
import argparse
import logging
import urllib.request
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import pcbnew
try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False
    logger.warning("pcbnew not available - run with KiCad's Python interpreter")

# Freerouting configuration
FREEROUTING_VERSION = "2.1.0"
FREEROUTING_URL = f"https://github.com/freerouting/freerouting/releases/download/v{FREEROUTING_VERSION}/freerouting-{FREEROUTING_VERSION}.jar"
FREEROUTING_JAR = f"freerouting-{FREEROUTING_VERSION}.jar"


@dataclass
class RoutingSettings:
    """Module-specific routing settings"""
    module_name: str
    max_passes: int = 100           # Maximum routing optimization passes
    trace_width_mm: float = 0.2     # Default trace width
    trace_clearance_mm: float = 0.2 # Minimum clearance
    via_diameter_mm: float = 0.6    # Via pad diameter
    via_drill_mm: float = 0.3       # Via drill diameter
    prefer_layer: str = "F.Cu"      # Preferred routing layer
    allow_vias: bool = True         # Allow vias for routing
    rf_mode: bool = False           # RF mode: wider traces, fewer vias

    @classmethod
    def for_module(cls, module_name: str) -> 'RoutingSettings':
        """Get default settings for a specific module type"""
        settings = cls(module_name=module_name)

        # RF matching network - controlled impedance, minimal vias
        if 'rf' in module_name.lower() or 'matching' in module_name.lower():
            settings.rf_mode = True
            settings.allow_vias = False
            settings.trace_width_mm = 0.3
            settings.max_passes = 50  # Fewer passes needed for simple RF

        # Power filter - wider traces
        elif 'power' in module_name.lower():
            settings.trace_width_mm = 0.4
            settings.trace_clearance_mm = 0.25

        # I2C/debug - standard digital
        elif 'i2c' in module_name.lower() or 'swd' in module_name.lower():
            settings.max_passes = 30  # Simple routing

        # Sensors/ICs - standard settings
        else:
            settings.max_passes = 60

        return settings


@dataclass
class RoutingResult:
    """Result of routing a module"""
    module_name: str
    success: bool
    routed_nets: int = 0
    unrouted_nets: int = 0
    vias: int = 0
    trace_length_mm: float = 0.0
    error_message: str = ""
    settings: Optional[RoutingSettings] = None


@dataclass
class ValidationIssue:
    """A single validation issue"""
    severity: str  # "error", "warning", "info"
    category: str  # "unconnected", "overlap", "placement", "netlist"
    message: str
    component: str = ""


class PreRoutingValidator:
    """Validates PCB before sending to autorouter

    Checks for common issues that cause routing failures:
    - Unconnected pads
    - Overlapping footprints
    - Components at origin (unplaced)
    - Missing or incomplete netlists
    """

    def __init__(self, board):
        self.board = board
        self.issues: List[ValidationIssue] = []

    def validate(self) -> List[ValidationIssue]:
        """Run all validation checks

        Returns:
            List of ValidationIssue objects
        """
        self.issues = []

        if not PCBNEW_AVAILABLE or not self.board:
            self.issues.append(ValidationIssue(
                severity="error",
                category="system",
                message="Board not loaded or pcbnew unavailable"
            ))
            return self.issues

        self._check_unconnected_pads()
        self._check_overlapping_footprints()
        self._check_unplaced_components()
        self._check_netlist_completeness()

        return self.issues

    def _check_unconnected_pads(self):
        """Check for pads without net assignments"""
        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            for pad in fp.Pads():
                net = pad.GetNet()
                pad_name = pad.GetNumber()

                # Skip pads that are intentionally unconnected (NC, mounting holes)
                if pad_name.upper() in ['NC', 'SHIELD', 'MP', 'MH']:
                    continue

                if not net or not net.GetNetname():
                    self.issues.append(ValidationIssue(
                        severity="warning",
                        category="unconnected",
                        message=f"Pad {pad_name} has no net assignment",
                        component=ref
                    ))

    def _check_overlapping_footprints(self):
        """Check for overlapping footprint pad areas

        Uses pad-based bounds instead of full bounding box to avoid
        false positives from silkscreen/courtyard overlaps.
        """
        footprints = list(self.board.GetFootprints())
        checked_pairs = set()

        for i, fp1 in enumerate(footprints):
            ref1 = fp1.GetReference()
            bbox1 = self._get_pad_bounds(fp1)
            if bbox1 is None:
                continue

            for fp2 in footprints[i+1:]:
                ref2 = fp2.GetReference()
                pair_key = tuple(sorted([ref1, ref2]))

                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                bbox2 = self._get_pad_bounds(fp2)
                if bbox2 is None:
                    continue

                # Check for intersection with clearance margin
                if self._boxes_overlap(bbox1, bbox2):
                    self.issues.append(ValidationIssue(
                        severity="error",
                        category="overlap",
                        message=f"Overlaps with {ref2}",
                        component=ref1
                    ))

    def _get_pad_bounds(self, fp):
        """Get bounding box based on pads only (tighter than full bounding box)"""
        pads = list(fp.Pads())
        if not pads:
            return None

        # Initialize with first pad
        first_pad = pads[0]
        pos = first_pad.GetPosition()
        size = first_pad.GetSize()

        min_x = pos.x - size.x // 2
        max_x = pos.x + size.x // 2
        min_y = pos.y - size.y // 2
        max_y = pos.y + size.y // 2

        # Expand to include all pads
        for pad in pads[1:]:
            pos = pad.GetPosition()
            size = pad.GetSize()

            min_x = min(min_x, pos.x - size.x // 2)
            max_x = max(max_x, pos.x + size.x // 2)
            min_y = min(min_y, pos.y - size.y // 2)
            max_y = max(max_y, pos.y + size.y // 2)

        # Create a BOX2I from the bounds
        bbox = pcbnew.BOX2I(
            pcbnew.VECTOR2I(min_x, min_y),
            pcbnew.VECTOR2I(max_x - min_x, max_y - min_y)
        )
        return bbox

    def _boxes_overlap(self, bbox1, bbox2) -> bool:
        """Check if two bounding boxes overlap with clearance margin"""
        # Use design rule clearance margin (0.2mm typical)
        margin = pcbnew.FromMM(0.15)

        return not (
            bbox1.GetRight() + margin < bbox2.GetLeft() or
            bbox1.GetLeft() - margin > bbox2.GetRight() or
            bbox1.GetBottom() + margin < bbox2.GetTop() or
            bbox1.GetTop() - margin > bbox2.GetBottom()
        )

    def _check_unplaced_components(self):
        """Check for components still at origin or with default placement"""
        origin_threshold = pcbnew.FromMM(1.0)  # Within 1mm of origin

        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            pos = fp.GetPosition()

            # Check if at origin
            if abs(pos.x) < origin_threshold and abs(pos.y) < origin_threshold:
                self.issues.append(ValidationIssue(
                    severity="warning",
                    category="placement",
                    message="Component appears to be at origin (unplaced?)",
                    component=ref
                ))

    def _check_netlist_completeness(self):
        """Check for basic netlist issues"""
        net_count = 0
        nets_with_single_pad = 0

        # Count nets and check for single-pad nets
        netinfo = self.board.GetNetInfo()
        for net in netinfo.NetsByNetcode():
            if net == 0:  # Skip unconnected net
                continue
            net_count += 1

        # Check for very low net count (possible netlist issue)
        footprint_count = len(list(self.board.GetFootprints()))
        if footprint_count > 2 and net_count < 2:
            self.issues.append(ValidationIssue(
                severity="error",
                category="netlist",
                message=f"Very few nets ({net_count}) for {footprint_count} components - netlist may be incomplete"
            ))

    def get_errors(self) -> List[ValidationIssue]:
        """Get only error-level issues"""
        return [i for i in self.issues if i.severity == "error"]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues"""
        return [i for i in self.issues if i.severity == "warning"]

    def has_blocking_issues(self) -> bool:
        """Check if there are any error-level issues that should block routing"""
        return len(self.get_errors()) > 0

    def print_report(self):
        """Print validation report to logger"""
        errors = self.get_errors()
        warnings = self.get_warnings()

        if not self.issues:
            logger.info("  Validation: All checks passed")
            return

        if errors:
            logger.warning(f"  Validation: {len(errors)} errors, {len(warnings)} warnings")
            for issue in errors:
                comp_str = f"[{issue.component}] " if issue.component else ""
                logger.error(f"    ERROR: {comp_str}{issue.message}")
        else:
            logger.info(f"  Validation: {len(warnings)} warnings (no blocking errors)")

        for issue in warnings[:5]:  # Limit warning output
            comp_str = f"[{issue.component}] " if issue.component else ""
            logger.warning(f"    WARN: {comp_str}{issue.message}")

        if len(warnings) > 5:
            logger.warning(f"    ... and {len(warnings) - 5} more warnings")


class FreeroutingManager:
    """Manages Freerouting JAR download and execution"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.tools_dir = os.path.join(project_dir, "tools")
        self.jar_path = os.path.join(self.tools_dir, FREEROUTING_JAR)

    def ensure_jar_available(self) -> bool:
        """Download Freerouting JAR if not present"""
        if os.path.exists(self.jar_path):
            logger.debug(f"Freerouting JAR exists: {self.jar_path}")
            return True

        logger.info(f"Downloading Freerouting {FREEROUTING_VERSION}...")
        os.makedirs(self.tools_dir, exist_ok=True)

        try:
            urllib.request.urlretrieve(FREEROUTING_URL, self.jar_path)
            logger.info(f"Downloaded: {self.jar_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download Freerouting: {e}")
            return False

    def check_java(self) -> bool:
        """Check if Java is available"""
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("Java not found. Please install JRE.")
            return False

    def run_freerouting(self, dsn_file: str, ses_file: str,
                        timeout: int = 300,
                        max_passes: int = 100,
                        headless: bool = True) -> Tuple[bool, str]:
        """Run Freerouting on a DSN file

        Args:
            dsn_file: Input Specctra DSN file
            ses_file: Output Specctra SES file
            timeout: Maximum time in seconds
            max_passes: Maximum routing passes
            headless: Run in headless mode (no GUI)

        Returns:
            Tuple of (success, error_message)
        """
        if not os.path.exists(self.jar_path):
            return False, "Freerouting JAR not found"

        if not os.path.exists(dsn_file):
            return False, f"DSN file not found: {dsn_file}"

        cmd = [
            "java", "-jar", self.jar_path,
            "-de", dsn_file,
            "-do", ses_file,
            "-mp", str(max_passes),  # Max passes
            "-dct", "0",              # Dialog confirmation timeout: 0 = immediate
        ]

        # Add headless mode options
        if headless:
            cmd.extend(["--gui.enabled", "false"])

        logger.info(f"Running Freerouting on {os.path.basename(dsn_file)}...")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 60,  # Extra buffer for startup/shutdown
                cwd=os.path.dirname(dsn_file)
            )

            if result.returncode == 0 and os.path.exists(ses_file):
                logger.info(f"  Routing complete: {os.path.basename(ses_file)}")
                return True, ""
            else:
                error = result.stderr or result.stdout or "Unknown error"
                logger.warning(f"  Freerouting failed: {error[:200]}")
                return False, error

        except subprocess.TimeoutExpired:
            return False, "Routing timed out"
        except Exception as e:
            return False, str(e)


class ModuleDSNExporter:
    """Exports module PCBs to DSN format"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.layouts_dir = os.path.join(project_dir, "layouts")

    def _clear_existing_traces(self, board) -> int:
        """Delete all existing traces and vias from the board

        Returns:
            Number of traces/vias deleted
        """
        deleted = 0

        # Collect tracks to delete (can't modify while iterating)
        tracks_to_delete = []
        for track in board.GetTracks():
            tracks_to_delete.append(track)

        # Delete tracks
        for track in tracks_to_delete:
            board.Remove(track)
            deleted += 1

        if deleted > 0:
            logger.info(f"  Deleted {deleted} existing traces/vias")

        return deleted

    def export_dsn(self, module_name: str, clear_traces: bool = True,
                   validate: bool = True, block_on_errors: bool = True) -> Tuple[bool, str]:
        """Export a module PCB to DSN format

        Args:
            module_name: Name of the module
            clear_traces: If True, delete existing traces before export
            validate: If True, run pre-routing validation
            block_on_errors: If True, fail on validation errors

        Returns:
            Tuple of (success, dsn_path or error_message)
        """
        if not PCBNEW_AVAILABLE:
            return False, "pcbnew not available"

        pcb_path = os.path.join(self.layouts_dir, module_name, f"{module_name}.kicad_pcb")
        dsn_path = os.path.join(self.layouts_dir, module_name, f"{module_name}.dsn")

        if not os.path.exists(pcb_path):
            return False, f"PCB not found: {pcb_path}"

        try:
            logger.info(f"Exporting DSN: {module_name}")

            # Run pre-routing validation
            if validate:
                board = pcbnew.LoadBoard(pcb_path)
                validator = PreRoutingValidator(board)
                validator.validate()
                validator.print_report()

                if block_on_errors and validator.has_blocking_issues():
                    errors = validator.get_errors()
                    error_msgs = [f"{e.component}: {e.message}" for e in errors[:3]]
                    del board
                    return False, f"Validation failed: {'; '.join(error_msgs)}"

                del board

            # Clear existing traces before exporting
            if clear_traces:
                board = pcbnew.LoadBoard(pcb_path)
                deleted = self._clear_existing_traces(board)
                if deleted > 0:
                    # Save the board with traces removed
                    board.Save(pcb_path)
                # Release the board reference to avoid SWIG issues
                del board

            # Load fresh board for DSN export
            board = pcbnew.LoadBoard(pcb_path)

            # Export to DSN
            success = pcbnew.ExportSpecctraDSN(board, dsn_path)

            # Clean up board reference to avoid SWIG issues
            del board

            if success and os.path.exists(dsn_path):
                logger.info(f"  Exported: {dsn_path}")
                return True, dsn_path
            else:
                return False, "ExportSpecctraDSN returned False"

        except Exception as e:
            return False, str(e)


class ModuleSESImporter:
    """Imports routed SES files back to PCBs"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.layouts_dir = os.path.join(project_dir, "layouts")

    def import_ses(self, module_name: str) -> Tuple[bool, str]:
        """Import SES routing results into a module PCB

        Returns:
            Tuple of (success, message)
        """
        if not PCBNEW_AVAILABLE:
            return False, "pcbnew not available"

        pcb_path = os.path.join(self.layouts_dir, module_name, f"{module_name}.kicad_pcb")
        ses_path = os.path.join(self.layouts_dir, module_name, f"{module_name}.ses")

        if not os.path.exists(pcb_path):
            return False, f"PCB not found: {pcb_path}"

        if not os.path.exists(ses_path):
            return False, f"SES not found: {ses_path}"

        # Backup original PCB
        backup_path = pcb_path + ".backup"
        shutil.copy(pcb_path, backup_path)

        board = None
        try:
            logger.info(f"Importing SES: {module_name}")

            # Load a fresh board for SES import
            board = pcbnew.LoadBoard(pcb_path)

            # Import SES
            success = pcbnew.ImportSpecctraSES(board, ses_path)

            if success:
                board.Save(pcb_path)
                # Clean up board reference before returning
                del board
                board = None
                logger.info(f"  Imported routing to: {pcb_path}")
                return True, f"Routing imported successfully"
            else:
                # Clean up board reference
                del board
                board = None
                # Restore backup
                shutil.copy(backup_path, pcb_path)
                return False, "ImportSpecctraSES returned False"

        except Exception as e:
            # Clean up board reference if it exists
            if board is not None:
                try:
                    del board
                except:
                    pass
            # Restore backup on error
            if os.path.exists(backup_path):
                shutil.copy(backup_path, pcb_path)
            return False, str(e)
        finally:
            # Clean up backup
            if os.path.exists(backup_path):
                os.remove(backup_path)


def _route_single_module_worker(args: Tuple[str, str, int, int]) -> Tuple[str, bool, str]:
    """Worker function to route a single module in a subprocess.

    This runs in a separate process to avoid SWIG wrapper corruption
    that occurs when pcbnew operates on multiple boards in sequence.

    Args:
        args: Tuple of (project_dir, module_name, timeout, max_passes)

    Returns:
        Tuple of (module_name, success, message)
    """
    project_dir, module_name, timeout, max_passes = args

    try:
        exporter = ModuleDSNExporter(project_dir)
        importer = ModuleSESImporter(project_dir)
        freerouting = FreeroutingManager(project_dir)

        # Step 1: Export DSN
        success, dsn_path = exporter.export_dsn(module_name)
        if not success:
            return (module_name, False, f"DSN export failed: {dsn_path}")

        # Step 2: Run Freerouting
        ses_path = dsn_path.replace(".dsn", ".ses")
        success, error = freerouting.run_freerouting(
            dsn_path, ses_path, timeout, max_passes
        )
        if not success:
            return (module_name, False, f"Routing failed: {error}")

        # Step 3: Import SES
        success, message = importer.import_ses(module_name)
        if not success:
            return (module_name, False, f"SES import failed: {message}")

        return (module_name, True, "Routed successfully")

    except Exception as e:
        return (module_name, False, str(e))


class ModuleAutorouter:
    """Orchestrates the complete autorouting workflow"""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.freerouting = FreeroutingManager(project_dir)
        self.exporter = ModuleDSNExporter(project_dir)
        self.importer = ModuleSESImporter(project_dir)

        # Module list from ato.yaml
        self.modules = self._get_modules()

    def _get_modules(self) -> List[str]:
        """Get list of module names from layouts directory"""
        layouts_dir = os.path.join(self.project_dir, "layouts")
        modules = []

        # Skip these non-module layouts
        skip_layouts = {'default', 'example'}

        for item in os.listdir(layouts_dir):
            if item in skip_layouts:
                continue
            layout_path = os.path.join(layouts_dir, item)
            if os.path.isdir(layout_path):
                pcb_file = os.path.join(layout_path, f"{item}.kicad_pcb")
                if os.path.exists(pcb_file):
                    modules.append(item)

        return sorted(modules)

    def route_module(self, module_name: str,
                     timeout: int = 300,
                     max_passes: int = 100) -> RoutingResult:
        """Route a single module using separate subprocesses to avoid SWIG issues.

        Runs export+routing in one subprocess, then import in a separate subprocess.

        Args:
            module_name: Name of the module
            timeout: Maximum routing time in seconds
            max_passes: Maximum routing passes

        Returns:
            RoutingResult with routing status
        """
        result = RoutingResult(module_name=module_name, success=False)
        script_path = os.path.abspath(__file__)

        # Step 1: Export DSN and run Freerouting in one subprocess
        cmd_export = [
            sys.executable,
            script_path,
            "--modules", module_name,
            "--timeout", str(timeout),
            "--passes", str(max_passes),
            "--single-worker",
            "--step", "export"  # Only export and route
        ]

        try:
            proc_result = subprocess.run(
                cmd_export,
                capture_output=True,
                text=True,
                timeout=timeout + 120,
                cwd=self.project_dir
            )

            if proc_result.returncode != 0:
                error_msg = self._parse_error(proc_result, module_name)
                result.error_message = error_msg
                return result

        except subprocess.TimeoutExpired:
            result.error_message = "Export subprocess timed out"
            return result
        except Exception as e:
            result.error_message = str(e)
            return result

        # Step 2: Import SES in a separate subprocess (clean pcbnew state)
        cmd_import = [
            sys.executable,
            script_path,
            "--modules", module_name,
            "--single-worker",
            "--step", "import"  # Only import
        ]

        try:
            proc_result = subprocess.run(
                cmd_import,
                capture_output=True,
                text=True,
                timeout=60,  # Import should be quick
                cwd=self.project_dir
            )

            if proc_result.returncode == 0:
                result.success = True
            else:
                error_msg = self._parse_error(proc_result, module_name)
                result.error_message = error_msg

        except subprocess.TimeoutExpired:
            result.error_message = "Import subprocess timed out"
        except Exception as e:
            result.error_message = str(e)

        return result

    def _parse_error(self, proc_result, module_name: str) -> str:
        """Parse error message from subprocess output"""
        error_msg = "Operation failed"
        combined_output = proc_result.stdout + proc_result.stderr
        for line in combined_output.splitlines():
            if "✗" in line and module_name in line:
                error_msg = line.strip()
                break
            elif "DSN export failed" in line or "SES import failed" in line or "Routing failed" in line:
                error_msg = line.strip()
                break
        return error_msg[:100]

    def route_module_direct(self, module_name: str,
                            timeout: int = 300,
                            max_passes: int = None,
                            step: str = "all") -> RoutingResult:
        """Route a single module directly (used by subprocess worker).

        Args:
            module_name: Name of the module
            timeout: Maximum routing time in seconds
            max_passes: Maximum routing passes (None = use module-specific)
            step: Which step to execute: "export", "import", or "all"

        Returns:
            RoutingResult with routing status
        """
        # Get module-specific routing settings
        settings = RoutingSettings.for_module(module_name)
        if max_passes is not None:
            settings.max_passes = max_passes

        result = RoutingResult(module_name=module_name, success=False, settings=settings)

        logger.info(f"  Settings: passes={settings.max_passes}, "
                    f"trace={settings.trace_width_mm}mm, rf_mode={settings.rf_mode}")

        layouts_dir = os.path.join(self.project_dir, "layouts")
        dsn_path = os.path.join(layouts_dir, module_name, f"{module_name}.dsn")
        ses_path = os.path.join(layouts_dir, module_name, f"{module_name}.ses")

        if step == "export" or step == "all":
            # Step 1: Export DSN
            success, export_result = self.exporter.export_dsn(module_name)
            if not success:
                result.error_message = f"DSN export failed: {export_result}"
                return result
            dsn_path = export_result

            # Step 2: Run Freerouting with module-specific settings
            success, error = self.freerouting.run_freerouting(
                dsn_path, ses_path, timeout, settings.max_passes, headless=True
            )
            if not success:
                result.error_message = f"Routing failed: {error}"
                return result

            if step == "export":
                result.success = True
                return result

        if step == "import" or step == "all":
            # Step 3: Import SES
            success, message = self.importer.import_ses(module_name)
            if not success:
                result.error_message = f"SES import failed: {message}"
                return result

        result.success = True
        return result

    def route_all(self, module_names: Optional[List[str]] = None,
                  timeout: int = 300,
                  max_passes: int = 100,
                  use_subprocess: bool = True) -> Dict[str, RoutingResult]:
        """Route multiple modules

        Args:
            module_names: List of modules to route, or None for all
            timeout: Maximum routing time per module
            max_passes: Maximum routing passes per module
            use_subprocess: If True, route each module in separate subprocess

        Returns:
            Dict mapping module name to RoutingResult
        """
        # Ensure Freerouting is available
        if not self.freerouting.check_java():
            logger.error("Java not available")
            return {}

        if not self.freerouting.ensure_jar_available():
            logger.error("Could not get Freerouting JAR")
            return {}

        # Determine which modules to route
        if module_names is None or "all" in module_names:
            modules_to_route = self.modules
        else:
            modules_to_route = [m for m in module_names if m in self.modules]

        logger.info(f"=== Routing {len(modules_to_route)} Modules ===")

        results = {}
        for module_name in modules_to_route:
            logger.info(f"\n--- Routing: {module_name} ---")

            if use_subprocess:
                result = self.route_module(module_name, timeout, max_passes)
            else:
                result = self.route_module_direct(module_name, timeout, max_passes)

            results[module_name] = result

            if result.success:
                logger.info(f"  ✓ {module_name}: Routed successfully")
            else:
                logger.warning(f"  ✗ {module_name}: {result.error_message}")

        return results

    def print_summary(self, results: Dict[str, RoutingResult]):
        """Print routing summary"""
        print(f"\n{'='*60}")
        print("Autorouting Summary")
        print(f"{'='*60}")

        success_count = sum(1 for r in results.values() if r.success)
        fail_count = len(results) - success_count

        print(f"Total modules: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        print()

        print("Results:")
        for name, result in sorted(results.items()):
            status = "✓ ROUTED" if result.success else "✗ FAILED"
            print(f"  {name:20} [{status}]")
            if not result.success:
                print(f"    Error: {result.error_message[:50]}")

        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Module Autorouter - Freerouting Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Route all modules
    python module_autorouter.py

    # Route specific modules
    python module_autorouter.py --modules power_filter,rf_matching

    # Route with custom timeout (5 minutes per module)
    python module_autorouter.py --timeout 300

    # Export DSN files only (no routing)
    python module_autorouter.py --export-only

    # Import SES files only (routing done externally)
    python module_autorouter.py --import-only

Note: Run this script with KiCad's Python interpreter:
    /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3 module_autorouter.py
"""
    )
    parser.add_argument('--modules', default='all',
                        help='Comma-separated list of modules to route, or "all"')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Maximum routing time per module in seconds (default: 300)')
    parser.add_argument('--passes', type=int, default=100,
                        help='Maximum routing passes (default: 100)')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export DSN files, do not route')
    parser.add_argument('--import-only', action='store_true',
                        help='Only import existing SES files')
    parser.add_argument('--single-worker', action='store_true',
                        help='Internal: run single module in subprocess mode')
    parser.add_argument('--step', choices=['export', 'import', 'all'], default='all',
                        help='Internal: which step to execute (export, import, all)')
    parser.add_argument('--no-subprocess', action='store_true',
                        help='Disable subprocess isolation (may cause SWIG errors)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not PCBNEW_AVAILABLE:
        logger.error("pcbnew module not available!")
        logger.error("Run with KiCad's Python interpreter:")
        logger.error("  /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3 module_autorouter.py")
        sys.exit(1)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    autorouter = ModuleAutorouter(project_dir)

    # Parse module names
    if args.modules == 'all':
        module_names = None
    else:
        module_names = [m.strip() for m in args.modules.split(',')]

    # Handle single-worker subprocess mode (internal use)
    if args.single_worker:
        if module_names is None or len(module_names) != 1:
            logger.error("--single-worker requires exactly one module")
            sys.exit(1)
        module = module_names[0]
        result = autorouter.route_module_direct(
            module, args.timeout, args.passes, step=args.step
        )
        if result.success:
            step_name = args.step.capitalize() if args.step != "all" else "Routed"
            logger.info(f"  ✓ {module}: {step_name} successfully")
            sys.exit(0)
        else:
            logger.error(f"  ✗ {module}: {result.error_message}")
            sys.exit(1)

    # Handle export-only mode
    if args.export_only:
        logger.info("=== Export DSN Files Only ===")
        for module in autorouter.modules if module_names is None else module_names:
            success, result = autorouter.exporter.export_dsn(module)
            status = "✓" if success else "✗"
            print(f"  {status} {module}: {result}")
        return

    # Handle import-only mode
    if args.import_only:
        logger.info("=== Import SES Files Only ===")
        for module in autorouter.modules if module_names is None else module_names:
            success, result = autorouter.importer.import_ses(module)
            status = "✓" if success else "✗"
            print(f"  {status} {module}: {result}")
        return

    # Full routing workflow (default: use subprocess isolation)
    use_subprocess = not args.no_subprocess
    results = autorouter.route_all(
        module_names=module_names,
        timeout=args.timeout,
        max_passes=args.passes,
        use_subprocess=use_subprocess
    )

    autorouter.print_summary(results)


if __name__ == "__main__":
    main()
