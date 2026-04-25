import os

# Multiple OpenMP runtimes are linked on this machine (common with conda + PyTorch on Windows).
# Must be set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
