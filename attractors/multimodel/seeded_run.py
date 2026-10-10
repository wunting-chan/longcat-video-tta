#!/usr/bin/env python3
"""Seed python/numpy/torch, then run a third-party script unmodified.

Usage: seeded_run.py SEED script.py [script args...]
Used for repos whose inference script does not seed (CausVid). The script runs as __main__
with sys.argv set to its own arguments, exactly as if launched directly.
"""
import random
import runpy
import sys

import numpy as np
import torch

seed = int(sys.argv[1])
script = sys.argv[2]
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
sys.argv = [script] + sys.argv[3:]
sys.path.insert(0, __import__("os").path.dirname(script))
runpy.run_path(script, run_name="__main__")
