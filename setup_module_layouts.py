#!/usr/bin/env python3
"""
Setup script for module layout directories.

This script creates the layout directories and fp-lib-table files
required for atopile module builds.
"""

import os
import re
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LAYOUTS_DIR = PROJECT_DIR / "layouts"
DEFAULT_LAYOUT = LAYOUTS_DIR / "default"

# All modules from ato.yaml
MODULES = [
    "power_filter",
    "i2c_bus",
    "lis3dhtr",
    "qmi8658c",
    "hdc2080",
    "eeprom",
    "rf_matching",
    "status_led",
    "ntc_temp",
    "swd_debug",
]


def create_module_fp_lib_table(module_name: str):
    """Create fp-lib-table for a module layout directory.

    Module layouts are at layouts/<module>/, which is one level deeper than
    layouts/default/. So paths need to be adjusted:
    - Default uses: ${KIPRJMOD}/../../parts/
    - Modules need: ${KIPRJMOD}/../../../parts/ (but that's wrong)

    Actually ${KIPRJMOD} refers to where the .kicad_pro is, which would be
    in the same directory as the PCB. So for layouts/power_filter/power_filter.kicad_pcb,
    ${KIPRJMOD}/../../parts/ resolves to layouts/power_filter/../../parts/ = parts/

    This is correct! We just need to copy the fp-lib-table as-is.
    """
    module_dir = LAYOUTS_DIR / module_name
    module_dir.mkdir(parents=True, exist_ok=True)

    source_fp_lib = DEFAULT_LAYOUT / "fp-lib-table"
    dest_fp_lib = module_dir / "fp-lib-table"

    if source_fp_lib.exists():
        shutil.copy(source_fp_lib, dest_fp_lib)
        print(f"  Created {dest_fp_lib.relative_to(PROJECT_DIR)}")
    else:
        # Create minimal fp-lib-table
        dest_fp_lib.write_text('(fp_lib_table\n  (version 7)\n)\n')
        print(f"  Created minimal {dest_fp_lib.relative_to(PROJECT_DIR)}")

    return module_dir


def create_module_kicad_pro(module_name: str):
    """Create a minimal .kicad_pro file for the module."""
    module_dir = LAYOUTS_DIR / module_name
    kicad_pro = module_dir / f"{module_name}.kicad_pro"

    # Minimal KiCad project file
    content = f'''{{
  "board": {{
    "design_settings": {{
      "defaults": {{
        "board_outline_line_width": 0.1,
        "copper_line_width": 0.2,
        "copper_text_size_h": 1.5,
        "copper_text_size_v": 1.5,
        "copper_text_thickness": 0.3,
        "other_line_width": 0.15,
        "silk_line_width": 0.15,
        "silk_text_size_h": 1.0,
        "silk_text_size_v": 1.0,
        "silk_text_thickness": 0.15
      }}
    }}
  }},
  "meta": {{
    "filename": "{module_name}.kicad_pro",
    "version": 1
  }},
  "text_variables": {{}}
}}
'''
    kicad_pro.write_text(content)
    print(f"  Created {kicad_pro.relative_to(PROJECT_DIR)}")


def create_empty_kicad_pcb(module_name: str):
    """Create an empty KiCad PCB file for the module."""
    module_dir = LAYOUTS_DIR / module_name
    pcb_path = module_dir / f"{module_name}.kicad_pcb"

    # Minimal KiCad PCB file (KiCad 9.0 format - version 20241229)
    content = '''(kicad_pcb
	(version 20241229)
	(generator "pcbnew")
	(generator_version "9.0")
	(general
		(thickness 1.6)
		(legacy_teardrops no)
	)
	(paper "A4")
	(layers
		(0 "F.Cu" signal)
		(2 "B.Cu" signal)
		(9 "F.Adhes" user "F.Adhesive")
		(11 "B.Adhes" user "B.Adhesive")
		(13 "F.Paste" user)
		(15 "B.Paste" user)
		(5 "F.SilkS" user "F.Silkscreen")
		(7 "B.SilkS" user "B.Silkscreen")
		(1 "F.Mask" user)
		(3 "B.Mask" user)
		(17 "Dwgs.User" user "User.Drawings")
		(19 "Cmts.User" user "User.Comments")
		(21 "Eco1.User" user "User.Eco1")
		(23 "Eco2.User" user "User.Eco2")
		(25 "Edge.Cuts" user)
		(27 "Margin" user)
		(31 "F.CrtYd" user "F.Courtyard")
		(29 "B.CrtYd" user "B.Courtyard")
		(35 "F.Fab" user)
		(33 "B.Fab" user)
		(39 "User.1" user)
		(41 "User.2" user)
		(43 "User.3" user)
		(45 "User.4" user)
		(47 "User.5" user)
		(49 "User.6" user)
		(51 "User.7" user)
		(53 "User.8" user)
		(55 "User.9" user)
	)
	(setup
		(pad_to_mask_clearance 0)
		(allow_soldermask_bridges_in_footprints no)
		(tenting front back)
		(pcbplotparams
			(layerselection 0x00000000_00000000_000010fc_ffffffff)
			(plot_on_all_layers_selection 0x00000000_00000000_00000000_00000000)
			(disableapertmacros no)
			(usegerberextensions no)
			(usegerberattributes yes)
			(usegerberadvancedattributes yes)
			(creategerberjobfile yes)
			(dashed_line_dash_ratio 12.000000)
			(dashed_line_gap_ratio 3.000000)
			(svgprecision 4)
			(plotframeref no)
			(mode 1)
			(useauxorigin no)
			(hpglpennumber 1)
			(hpglpenspeed 20)
			(hpglpendiameter 15.000000)
			(pdf_front_fp_property_popups yes)
			(pdf_back_fp_property_popups yes)
			(pdf_metadata yes)
			(pdf_single_document no)
			(dxfpolygonmode yes)
			(dxfimperialunits yes)
			(dxfusepcbnewfont yes)
			(psnegative no)
			(psa4output no)
			(plot_black_and_white yes)
			(sketchpadsonfab no)
			(plotpadnumbers no)
			(hidednponfab no)
			(sketchdnponfab yes)
			(crossoutdnponfab yes)
			(subtractmaskfromsilk no)
			(outputformat 1)
			(mirror no)
			(drillshape 1)
			(scaleselection 1)
			(outputdirectory "")
		)
	)
	(net 0 "")
)
'''
    pcb_path.write_text(content)
    print(f"  Created {pcb_path.relative_to(PROJECT_DIR)}")


def main():
    print("Setting up module layout directories...")
    print(f"Project: {PROJECT_DIR}")
    print(f"Layouts: {LAYOUTS_DIR}")
    print()

    for module_name in MODULES:
        print(f"Module: {module_name}")
        create_module_fp_lib_table(module_name)
        create_module_kicad_pro(module_name)
        create_empty_kicad_pcb(module_name)
        print()

    print("Done!")
    print(f"\nCreated {len(MODULES)} module layout directories.")
    print("\nNext steps:")
    print("  1. Run: ato build --target <module_name>  (for each module)")
    print("  2. Or run: python module_placer.py")


if __name__ == "__main__":
    main()
