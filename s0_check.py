import os
import sys
import importlib
import fi_config as C

print("Python", sys.version.split()[0])
ok = True
for name in ["numpy", "scipy", "pandas", "sklearn", "matplotlib"]:
    try:
        m = importlib.import_module(name)
        print("OK ", name, m.__version__)
    except Exception as e:
        ok = False
        print("MISSING", name, e)

try:
    import openpyxl
    print("OK  openpyxl", openpyxl.__version__)
except Exception:
    print("openpyxl missing (only needed for step 9): pip install openpyxl")

try:
    import torch
    print("OK  torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("    GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch not available (only needed for step 4):", e)

for sub in ["train/tdcsfog", "train/defog", "events.csv", "tasks.csv", "tdcsfog_metadata.csv", "defog_metadata.csv"]:
    p = os.path.join(C.KAGGLE_ROOT, sub)
    exists = os.path.exists(p)
    ok = ok and exists
    extra = ""
    if exists and os.path.isdir(p):
        extra = "(%d files)" % len([f for f in os.listdir(p) if f.endswith(".csv")])
    print("OK " if exists else "NOT FOUND", p, extra)

p = C.DAPHNET_ROOT
if os.path.isdir(p):
    print("OK ", p, "(%d files)" % len([f for f in os.listdir(p) if f.endswith(".txt")]))
else:
    print("NOT FOUND (Daphnet will be skipped)", p)

print("\nALL GOOD" if ok else "\nFIX THE ITEMS ABOVE BEFORE STEP 1")
