import os, platform, sys

print("HOST:", platform.node())
print("OS:", platform.system(), platform.machine())
print("PYTHON:", sys.executable)
print("CWD:", os.getcwd())
