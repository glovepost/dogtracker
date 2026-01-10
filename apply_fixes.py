import os

# Fix 1: Resolve Config Ambiguity in configuration_utils.py
# Prioritize 'text_config' if available, otherwise take the first one.
tgt1 = "tools/venv/lib/python3.14/site-packages/transformers/configuration_utils.py"
if os.path.exists(tgt1):
    print(f"Patching {tgt1}...")
    with open(tgt1, "r") as f: lines = f.readlines()
    with open(tgt1, "w") as f:
        for line in lines:
            if "PCT_PATCH" in line:
                continue # Remove old patch
            if "if len(valid_text_config_names) > 1:" in line:
                indent = line.split("if")[0]
                f.write(f"{indent}# Priority fix: Prefer text_config\n")
                f.write(f"{indent}if 'text_config' in valid_text_config_names: valid_text_config_names = ['text_config']\n")
                f.write(f"{indent}elif len(valid_text_config_names) > 1: valid_text_config_names = [valid_text_config_names[0]]\n")
                f.write(line) # Write original check (which will now pass)
            else:
                f.write(line)

# Fix 2: Handle Dict in Generation Config in generation/configuration_utils.py
# Sometimes config is returned as dict, so .to_dict() fails.
tgt2 = "tools/venv/lib/python3.14/site-packages/transformers/generation/configuration_utils.py"
if os.path.exists(tgt2):
    print(f"Patching {tgt2}...")
    with open(tgt2, "r") as f: lines = f.readlines()
    with open(tgt2, "w") as f:
        for line in lines:
            if "decoder_config_dict = decoder_config.to_dict()" in line:
                 indent = line.split("decoder")[0]
                 f.write(f"{indent}if isinstance(decoder_config, dict): decoder_config_dict = decoder_config\n")
                 f.write(f"{indent}else: decoder_config_dict = decoder_config.to_dict()\n")
            else:
                f.write(line)
