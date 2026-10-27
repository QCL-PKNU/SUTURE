"""Repo-relative sys.path setup shared by experiments/ and tests/.

Adds suture/lib and the upstream rasengan clone (../rasengan) to sys.path.
Run scripts from the suture/ root, e.g.:  python experiments/pilot_a.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
SUTURE_ROOT = os.path.dirname(_HERE)
REPO_ROOT = os.path.dirname(SUTURE_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "rasengan"))
sys.path.insert(0, os.path.join(SUTURE_ROOT, "lib"))
