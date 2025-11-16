# interventions/evaluate/report.py
import os, json
from datetime import datetime

def save_json(obj, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{name}.json")
    with open(fn, "w") as f:
        json.dump(obj, f, indent=2)
    return fn

def stamp_dir(base="/sc-cbint-vol/vlgcbm-outputs/interventions"):
    d = os.path.join(base, f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d
