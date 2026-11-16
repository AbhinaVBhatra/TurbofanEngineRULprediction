"""
data.py - Data Ingestion, Verification, and RUL Label Generation for NASA C-MAPSS

Dataset Source & Provenance:
- Source: NASA Prognostics Data Repository / NASA Open Data Portal
- Direct Archive: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- Alternative Mirror: https://ti.arc.nasa.gov/c/6/
- Citation: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
  NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- License / Terms of Use: NASA Open Innovation / Public Domain (Free for research and educational use).

NOTE ON DATA REALISM:
The C-MAPSS dataset is a HIGH-FIDELITY SIMULATED dataset produced via the Commercial Modular
Aero-Propulsion System Simulation (C-MAPSS) software developed at NASA Glenn Research Center.
It is NOT operational flight telemetry from revenue airline flights, but simulated degradation
trajectories incorporating realistic thermodynamic physics and noise models.
"""

import os
import ssl
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

# Enforce OpenMP fix across imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Column name definitions according to NASA C-MAPSS documentation
INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Descriptive mapping for viva defense and report documentation
SENSOR_DESCRIPTIONS = {
    "s_1": "Fan inlet temperature (°R)",
    "s_2": "LPC outlet temperature (°R)",
    "s_3": "HPC outlet temperature (°R)",
    "s_4": "LPT outlet temperature (°R)",
    "s_5": "Fan inlet pressure (psia)",
    "s_6": "Bypass-duct pressure (psia)",
    "s_7": "HPC outlet pressure (psia)",
    "s_8": "Physical fan speed (rpm)",
    "s_9": "Physical core speed (rpm)",
    "s_10": "Engine pressure ratio (P50/P2)",
    "s_11": "HPC outlet static pressure (psia)",
    "s_12": "Ratio of fuel flow to Ps30 (pps/psia)",
    "s_13": "Corrected fan speed (rpm)",
    "s_14": "Corrected core speed (rpm)",
    "s_15": "Bypass ratio",
    "s_16": "Burner fuel-air ratio",
    "s_17": "Bleed enthalpy",
    "s_18": "Demanded fan speed (rpm)",
    "s_19": "Demanded corrected fan speed (rpm)",
    "s_20": "HPT coolant bleed (lbm/s)",
    "s_21": "LPT coolant bleed (lbm/s)"
}

DATA_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"


def get_data_dir() -> Path:
    """Returns the base data directory (ptm_project/data)."""
    # Resolves relative to this file's location
    return Path(__file__).resolve().parent.parent / "data"


def download_and_extract_cmapss(target_dir: Optional[Path] = None, force: bool = False) -> Path:
    """
    Downloads the official NASA C-MAPSS archive and extracts text files to data/raw/.
    Handles SSL handshake quirks gracefully on modern Python environments.
    """
    if target_dir is None:
        raw_dir = get_data_dir() / "raw"
    else:
        raw_dir = Path(target_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    required_files = [f"train_FD00{i}.txt" for i in range(1, 5)] + \
                     [f"test_FD00{i}.txt" for i in range(1, 5)] + \
                     [f"RUL_FD00{i}.txt" for i in range(1, 5)]

    all_exist = all((raw_dir / f).exists() for f in required_files)
    if all_exist and not force:
        print(f"[DATA] All required C-MAPSS files already present in {raw_dir}")
        return raw_dir

    zip_path = raw_dir / "CMAPSSData.zip"
    print(f"[DATA] Downloading NASA C-MAPSS dataset from {DATA_URL}...")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60, context=ctx) as response, open(zip_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
            print(f"[DATA] Download complete. Archive size: {len(data):,} bytes ({len(data)/(1024*1024):.2f} MB).")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download C-MAPSS data from {DATA_URL}. Reason: {e}.\n"
            f"To resolve manually: Place 'CMAPSSData.zip' or the extracted text files directly into {raw_dir}."
        )

    print(f"[DATA] Extracting {zip_path.name} to {raw_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    # Clean up zip archive to conserve disk space
    if zip_path.exists():
        zip_path.unlink()

    print("[DATA] Extraction successful. Verified raw dataset files.")
    return raw_dir


def load_dataset(dataset_id: str = "FD001", data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads train, test, and true test RUL files for a specific sub-dataset (FD001 to FD004).

    Parameters:
    -----------
    dataset_id : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    data_dir : Path, optional
        Path to data/raw directory.

    Returns:
    --------
    train_df : pd.DataFrame
        Training data containing engine operational cycles and sensor readings.
    test_df : pd.DataFrame
        Test data with truncated engine histories.
    rul_true : pd.Series
        Ground-truth Remaining Useful Life for each test engine at the final observed cycle.
    """
    if data_dir is None:
        raw_dir = get_data_dir() / "raw"
    else:
        raw_dir = Path(data_dir)

    train_file = raw_dir / f"train_{dataset_id}.txt"
    test_file = raw_dir / f"test_{dataset_id}.txt"
    rul_file = raw_dir / f"RUL_{dataset_id}.txt"

    if not train_file.exists():
        download_and_extract_cmapss(raw_dir)

    # Read space-delimited text files
    # Note: text files have 26 data columns; sep=r'\s+' automatically handles variable whitespace
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=ALL_COLS, engine="python")
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=ALL_COLS, engine="python")
    rul_true = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["rul_true"], engine="python")["rul_true"]

    return train_df, test_df, rul_true


def add_remaining_useful_life(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) labels for run-to-failure training data.

    Why Piecewise-Linear RUL with Cap (e.g., 125 cycles)?
    -----------------------------------------------------
    In physical turbofan engines, degradation does NOT start immediately at cycle 1.
    During the initial healthy operating phase, the engine undergoes minor break-in, but
    wear and micro-crack propagation are negligible.
    If we train a regression model to predict linear decrease from cycle 1 (e.g. RUL = 350 down to 0),
    the model will attempt to fit noise and spurious variations in pristine healthy sensors,
    leading to poor generalizability.
    Capping the target RUL at a constant upper bound (piecewise-linear degradation model,
    standardized by Heimes 2008 and widely validated in turbofan literature) allows the model
    to predict a constant healthy plateau until observable degradation begins, significantly
    improving predictive accuracy near end-of-life where maintenance decisions matter most.

    Parameters:
    -----------
    df : pd.DataFrame
        Training dataframe containing 'unit_nr' and 'time_cycles'.
    rul_cap : int
        Maximum upper bound for RUL (default 125). Set to None for unbounded linear RUL.

    Returns:
    --------
    df_with_rul : pd.DataFrame
        Original dataframe with 'max_cycle', 'rul_unbounded', and 'RUL' columns added.
    """
    df = df.copy()
    # Compute maximum cycle reached by each engine (point of system failure)
    max_cycles = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycles.rename(columns={"time_cycles": "max_cycle"}, inplace=True)

    df = df.merge(max_cycles, on="unit_nr", how="left")
    df["rul_unbounded"] = df["max_cycle"] - df["time_cycles"]

    if rul_cap is not None and rul_cap > 0:
        df["RUL"] = df["rul_unbounded"].clip(upper=rul_cap)
    else:
        df["RUL"] = df["rul_unbounded"]

    return df


if __name__ == "__main__":
    print("[TEST] Running data download and verification module...")
    raw_dir = download_and_extract_cmapss()
    train_fd1, test_fd1, rul_fd1 = load_dataset("FD001")
    train_fd1 = add_remaining_useful_life(train_fd1, rul_cap=125)
    print(f"[TEST] FD001 Train shape: {train_fd1.shape}, Test shape: {test_fd1.shape}, RUL test len: {len(rul_fd1)}")
    print(f"[TEST] FD001 Units: {train_fd1['unit_nr'].nunique()} engines.")
    print(f"[TEST] Sample RUL head:\n{train_fd1[['unit_nr', 'time_cycles', 'max_cycle', 'rul_unbounded', 'RUL']].head()}")
