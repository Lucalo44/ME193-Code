#!/usr/bin/env python3
"""
Copy gaitid/ (+ trained models) into app/python/ so the app folder is
self-contained, then print the deploy command. Run after every train or edit
to gaitid/.
"""

import glob
import os
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dst_pkg = os.path.join(HERE, "app", "python", "gaitid")
dst_models = os.path.join(HERE, "app", "python", "models")
for d in (dst_pkg, dst_models):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
for f in ("__init__", "config", "features", "model"):
    shutil.copy(os.path.join(HERE, "gaitid", f + ".py"), dst_pkg)
for f in glob.glob(os.path.join(HERE, "models", "*.json")):
    shutil.copy(f, dst_models)
board_dir = os.path.join(HERE, "..", "ArduinoUNOq")
if os.path.isdir(board_dir):
    target = os.path.join(board_dir, "gaitid")
    shutil.copytree(os.path.join(HERE, "app"), target, dirs_exist_ok=True)
    print(f"Copied app/ -> {os.path.normpath(target)}")
    print("Deploy with:  python3 ArduinoUNOq/tools/deploy.py gaitid")
else:
    print("Synced app/. Copy it into your ArduinoUNOq folder as 'gaitid' and deploy it.")
