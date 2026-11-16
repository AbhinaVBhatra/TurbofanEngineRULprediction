"""
Turbofan Engine Remaining Useful Life (RUL) Prediction & Maintenance Decision Support
Source package initialization.
"""
import os

# Mitigate Intel OpenMP dual-runtime crash on Windows environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
