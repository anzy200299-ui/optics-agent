"""Matched-cost photonic inverse-design benchmark."""

import os

# One thread per process. Workers are separate processes; a threaded BLAS
# inside each of them oversubscribes the machine and slows the sparse solves.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
