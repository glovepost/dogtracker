import pandas as pd
import os

bom_path = "RAK2270-Sticker-Tracker/Hardware/BOM of RAK2270(FPCB-WITHOUT EEPROM-OPENSOURCE).xls"
output_path = "RAK2270-Sticker-Tracker/Hardware/BOM.md"

try:
    # Read as headerless first to find the real header row dynamically
    df_raw = pd.read_excel(bom_path, engine='xlrd', header=None)
    
    # Find the row index that contains "RAK code"
    # match rows where any cell equals "RAK code"
    header_matches = df_raw.index[df_raw.apply(lambda row: row.astype(str).str.contains("RAK code", case=False).any(), axis=1)].tolist()
    
    if not header_matches:
        print("Could not find header row containing 'RAK code'")
        exit(1)
    
    header_idx = header_matches[0]
    
    # Slice dataframe: Data starts after header_idx
    df = df_raw.iloc[header_idx+1:].copy()
    # Set columns from the identified header row
    df.columns = df_raw.iloc[header_idx].values
    
    # Clean headers: convert to string, strip whitespace, remove newlines
    df.columns = [str(c).strip().replace('\n', ' ') for c in df.columns]
    
    # Drop rows that are fully empty
    df = df.dropna(how='all')
    
    # Drop rows where 'RAK code' is missing or empty string
    # Try/Except in case column name isn't exactly 'RAK code' (e.g. whitespace)
    rak_col = [c for c in df.columns if 'RAK code' in c][0]
    df = df[df[rak_col].notna()]
    df = df[df[rak_col].astype(str).str.strip() != '']
    
    # Replace NaN values with empty string for cleaner markdown
    df = df.fillna("")

    # Convert to markdown
    markdown_table = df.to_markdown(index=False)

    with open(output_path, "w") as f:
        f.write("# RAK2270 Sticker Tracker BOM\n\n")
        f.write(markdown_table)
    
    print(f"Successfully converted BOM to {output_path}")

except Exception as e:
    print(f"Error converting BOM: {e}")
    import traceback
    traceback.print_exc()
