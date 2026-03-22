import sys
import traceback

print("Testing api.main import...")
try:
    from api.main import app
    print("api.main imported OK")
except Exception as e:
    print(f"api.main FAILED:")
    traceback.print_exc()
