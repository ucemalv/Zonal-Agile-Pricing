import yaml

# Load yaml
with open('environment.yaml', 'r') as f:
    env = yaml.safe_load(f)

# Extract dependencies
dependencies = env['dependencies']

# Filter out only the pip packages
pip_packages = []
for dep in dependencies:
    if isinstance(dep, str):
        pip_packages.append(dep)
    elif isinstance(dep, dict) and 'pip' in dep:
        pip_packages.extend(dep['pip'])

with open('requirements.txt', 'w') as f:
    for package in pip_packages:
        f.write(f"{package}\n")
