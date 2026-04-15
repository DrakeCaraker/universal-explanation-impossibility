#!/usr/bin/env python3
"""Wrapper to run mech_interp_rashomon_gpu.py on SageMaker."""
import sys
sys.path.insert(0, '.')
import mech_interp_rashomon_gpu as mi
mi.main()
