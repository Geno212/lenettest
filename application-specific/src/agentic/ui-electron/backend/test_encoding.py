"""Quick test to verify UTF-8 encoding works"""
import sys
import os

# Set UTF-8 encoding for Windows - using reconfigure method
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Test printing emojis
print("Testing emoji output:")
print("✅ Check mark")
print("❌ Cross mark")
print("🤖 Robot")
print("📝 Memo")
print("🧵 Thread")

print("\nIf you see these emojis without errors, UTF-8 encoding works!")
