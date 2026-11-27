"""On-board perception (keyword spotting, pose, beat tracking). Importing this package
before numpy keeps BLAS on one thread: numpy's default spreads one matrix multiply over all
four cores, which made the planner 2.7x slower (docs/SPIKES.md, Part 2 A0)."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
