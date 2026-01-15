"""Test that only JSON goes to stdout"""
import subprocess
import json
import sys

bridge_path = r"d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer\src\agentic\ui-electron\backend\chatbot_bridge.py"
python_path = r"d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer\venv\Scripts\python.exe"

# Start the bridge
proc = subprocess.Popen(
    [python_path, "-u", bridge_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='utf-8',
    cwd=r"d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer"
)

print("Bridge started, waiting for initialization...")

# Read lines from stdout (should be JSON only)
import select
import time

time.sleep(3)  # Wait for initialization

# Check if there's any output
try:
    # Read stdout line by line
    for i in range(10):  # Try to read up to 10 lines
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line:
            print(f"STDOUT Line {i+1}: {line[:100]}...")
            try:
                msg = json.loads(line)
                print(f"  ✅ Valid JSON: {msg.get('type')}")
            except json.JSONDecodeError as e:
                print(f"  ❌ Invalid JSON: {e}")
except Exception as e:
    print(f"Error reading: {e}")

# Terminate
proc.terminate()
proc.wait()

print("\nTest complete!")
