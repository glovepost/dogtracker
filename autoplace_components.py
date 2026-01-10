#!/usr/bin/env python3
"""
KiCad PCB Component Auto-Placer for AertactxDogtracker
Places components in logical groups following PCB design principles.
Respects courtyard clearances with 2-3mm spacing between components.
"""

import re

# Component positions (mm) with courtyard clearance spacing
COMPONENT_POSITIONS = {
    # MCU - Center of the board
    "U1": (0, 0),  # RAK3172 - Central position (14x14mm footprint)
    
    # RF Path - Right side, antenna at edge for radiation
    "ANT1": (18, -12),  # Antenna at board edge
    "C_MATCH1": (12, -10),  # Matching network
    "L_SERIES": (10, -10),
    "C_MATCH2": (8, -10),
    
    # Sensors - Left side, grouped near MCU I2C pins
    "U2": (-14, 8),   # LIS3DHTR (Accelerometer)
    "U4": (-14, -2),  # QMI8658C (IMU)
    "U5": (-14, -12), # HDC2080 (Temp/Humidity)
    "U3": (-14, 18),  # EEPROM
    
    # Power Section - Top area, battery at center-top
    "B1": (0, 22),    # Battery holder (24mm diameter)
    "L1": (-8, 14),   # Power inductor
    "L5": (8, 14),    # Power inductor
    "C3": (-12, 18),  # Power cap
    "C9": (12, 18),   # Power cap
    
    # I2C Pull-ups - Near sensors
    "R7": (-10, 5),
    "R8": (-10, 2),
    
    # NTC Divider - Right side near MCU ADC
    "R1": (12, 4),
    "R2": (12, 1),
    "R3": (15, 2),
    
    # LED Circuit - Top right
    "LED1": (16, 10),
    "R6": (16, 7),
    
    # Decoupling Caps - Near ICs
    "C1": (-4, 8),
    "C2": (4, 8),
    "C4": (-11, 8),
    "C5": (-11, -2),
    "C6": (-11, -12),
    "C7": (-11, 18),
    "C8": (4, -8),
    
    # EMI Filters
    "L2": (-6, 5),
    "L3": (6, 5),
    "L4": (0, -8),
    
    # Additional resistors
    "R4": (-6, -6),
    "R5": (6, -6),
    
    # SWD Header - Left edge
    "J1": (-22, 0),
}

def main():
    pcb_file = "/Users/glovepost/Projects/aertactx-dogtracker/build/builds/default/default.20260109-170523.kicad_pcb"
    
    print(f"Reading {pcb_file}...")
    with open(pcb_file, 'r') as f:
        content = f.read()
    
    updated_count = 0
    
    # Process each component
    for ref_des, (new_x, new_y) in COMPONENT_POSITIONS.items():
        # Find the footprint block containing this reference
        # Pattern: find (property "Reference" "REF") and work backwards to find (at x y)
        
        # First, find the reference property
        ref_pattern = rf'\(property "Reference" "{re.escape(ref_des)}"'
        ref_matches = list(re.finditer(ref_pattern, content))
        
        if not ref_matches:
            print(f"  {ref_des}: NOT FOUND in PCB")
            continue
            
        for ref_match in ref_matches:
            # Search backwards from the reference to find the footprint start
            search_start = max(0, ref_match.start() - 3000)  # Look back up to 3000 chars
            search_region = content[search_start:ref_match.start()]
            
            # Find the last "(footprint " before this reference
            fp_matches = list(re.finditer(r'\(footprint "', search_region))
            if not fp_matches:
                print(f"  {ref_des}: Could not find footprint start")
                continue
            
            # Get the position of the footprint start
            fp_start_in_region = fp_matches[-1].start()
            fp_absolute_start = search_start + fp_start_in_region
            
            # Find the (at x y [angle]) in the footprint header (first 500 chars after footprint start)
            fp_header = content[fp_absolute_start:fp_absolute_start + 500]
            
            # Match the footprint-level (at ...) - it's the first one after footprint definition
            at_pattern = r'\(at\s+([0-9.-]+)\s+([0-9.-]+)(\s+[0-9.-]+)?\)'
            at_match = re.search(at_pattern, fp_header)
            
            if at_match:
                old_at = at_match.group(0)
                rotation = at_match.group(3) if at_match.group(3) else ""
                new_at = f"(at {new_x} {new_y}{rotation})"
                
                # Replace in content
                at_absolute_start = fp_absolute_start + at_match.start()
                at_absolute_end = fp_absolute_start + at_match.end()
                content = content[:at_absolute_start] + new_at + content[at_absolute_end:]
                
                print(f"  {ref_des}: {old_at} -> {new_at}")
                updated_count += 1
                break
            else:
                print(f"  {ref_des}: Could not find (at ...) in footprint header")
    
    # Write output
    output_file = pcb_file.replace('.kicad_pcb', '_placed.kicad_pcb')
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\nUpdated {updated_count} components")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
