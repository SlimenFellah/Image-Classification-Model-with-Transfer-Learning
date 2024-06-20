import subprocess
import sys

# Read the requirements file
with open('requirements.txt', 'r') as file:
    packages = file.readlines()

# Install each package individually
for package in packages:
    package = package.strip()
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}. Skipping...")

print("Installation complete.")
