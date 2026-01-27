import os
import subprocess
import sys

folder = r".\test\ops"
for filename in os.listdir(folder):
    if filename.endswith(".py"):
        filepath = os.path.join(folder, filename)
        print(f"Running {filename}...")
        result = subprocess.run([sys.executable, filepath])
        if result.returncode != 0:
            print(f"⚠️  Error in {filename}")