"""
provenance_tracker.py - Provenance Registry for Experimental Numbers

Maps every numerical claim to the exact script, dataset, configuration, and output file.
Enforces Rule 2: NEVER invent numbers.
"""

import os
import csv
from pathlib import Path
from typing import Union, Optional

PROVENANCE_CSV = Path(__file__).resolve().parent.parent / "paper" / "number_provenance.csv"

FIELDNAMES = [
    "metric_id",
    "dataset",
    "model",
    "metric_name",
    "value",
    "unit",
    "script_source",
    "output_file",
    "notes"
]


def log_number(
    metric_id: str,
    dataset: str,
    model: str,
    metric_name: str,
    value: Union[float, int, str],
    unit: str,
    script_source: str,
    output_file: str,
    notes: str = ""
):
    """Appends or updates a record in paper/number_provenance.csv."""
    os.makedirs(PROVENANCE_CSV.parent, exist_ok=True)
    
    records = []
    file_exists = PROVENANCE_CSV.exists()
    
    if file_exists:
        with open(PROVENANCE_CSV, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
            
    # Check if metric_id already exists to avoid exact duplicate rows
    updated = False
    for r in records:
        if r["metric_id"] == metric_id:
            r["dataset"] = dataset
            r["model"] = model
            r["metric_name"] = metric_name
            r["value"] = str(value)
            r["unit"] = unit
            r["script_source"] = script_source
            r["output_file"] = output_file
            r["notes"] = notes
            updated = True
            break
            
    if not updated:
        records.append({
            "metric_id": metric_id,
            "dataset": dataset,
            "model": model,
            "metric_name": metric_name,
            "value": str(value),
            "unit": unit,
            "script_source": script_source,
            "output_file": output_file,
            "notes": notes
        })
        
    with open(PROVENANCE_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
