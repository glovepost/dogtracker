
path = 'tools/venv/lib/python3.14/site-packages/transformers/configuration_utils.py'
with open(path) as f: lines = f.readlines()
new_lines = []
for line in lines:
    if 'if len(valid_text_config_names) > 1:' in line:
        indent = line[:line.find('if')]
        # Force the list to contain only the first element if multiple are found,
        # effectively resolving the ambiguity by picking the first one.
        new_lines.append(f'{indent}if len(valid_text_config_names) > 1: valid_text_config_names = [valid_text_config_names[0]] # PCT_PATCH\n')
    new_lines.append(line)
with open(path, 'w') as f: f.writelines(new_lines)
