"""
Helper functions for Medicare Advantage data processing.
Python equivalent of functions.R
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

MA_DATA_DIR = Path("/home/gdanzil/econ470/a0/work/ma-data")


def read_contract(path: str) -> pd.DataFrame:
    """Read contract info CSV file."""
    col_names = [
        "contractid", "planid", "org_type", "plan_type", "partd", "snp", "eghp",
        "org_name", "org_marketing_name", "plan_name", "parent_org",
        "contract_date",
    ]
    df = pd.read_csv(
        path,
        skiprows=1,
        names=col_names,
        encoding="latin1", 
        encoding_errors="replace",
        dtype={
            "contractid": str,
            "planid": float,
            "org_type": str,
            "plan_type": str,
            "partd": str,
            "snp": str,
            "eghp": str,
            "org_name": str,
            "org_marketing_name": str,
            "plan_name": str,
            "parent_org": str,
            "contract_date": str,
        },
    )
    return df


def read_enroll(path: str) -> pd.DataFrame:
    """Read enrollment info CSV file."""
    col_names = ["contractid", "planid", "ssa", "fips", "state", "county", "enrollment"]
    df = pd.read_csv(
        path,
        skiprows=1,
        names=col_names,
        na_values="*",
        encoding="latin1", 
        encoding_errors="replace",
        dtype={
            "contractid": str,
            "planid": float,
            "ssa": float,
            "fips": float,
            "state": str,
            "county": str,
            "enrollment": float,
        },
    )
    return df


def load_month(m: str, y: int) -> pd.DataFrame:
    """Load one month of plan/enrollment data."""
    enrollment_dir = MA_DATA_DIR / "ma" / "enrollment" / "Extracted Data"

    c_path = enrollment_dir / f"CPSC_Contract_Info_{y}_{m}.csv"
    e_path = enrollment_dir / f"CPSC_Enrollment_Info_{y}_{m}.csv"

    if not c_path.exists():
        raise FileNotFoundError(f"Missing contract file: {c_path}")
    if not e_path.exists():
        raise FileNotFoundError(f"Missing enrollment file: {e_path}")

    contract_info = read_contract(c_path).drop_duplicates(subset=["contractid", "planid"], keep="first")
    enroll_info = read_enroll(e_path)

    merged = contract_info.merge(enroll_info, on=["contractid", "planid"], how="left")
    merged["month"] = int(m)
    merged["year"] = y
    return merged



def read_service_area(path: str) -> pd.DataFrame:
    """Read service area CSV file."""
    col_names = [
        "contractid", "org_name", "org_type", "plan_type", "partial", "eghp",
        "ssa", "fips", "county", "state", "notes",
    ]
    df = pd.read_csv(
        path,
        skiprows=1,
        names=col_names,
        encoding="latin1", 
        encoding_errors="replace",
        na_values="*",
        dtype={
            "contractid": str,
            "org_name": str,
            "org_type": str,
            "plan_type": str,
            "partial": str,
            "eghp": str,
            "ssa": float,
            "fips": float,
            "county": str,
            "state": str,
            "notes": str,
        },
    )

    # Convert partial to boolean (handles TRUE/FALSE strings and booleans)
    df["partial"] = df["partial"].map({"TRUE": True, "FALSE": False, True: True, False: False})
    return df


def load_month_sa(m: str, y: int) -> pd.DataFrame:
    """Load one month of service area data."""
    path = MA_DATA_DIR / "ma" / "service-area" / "Extracted Data" / f"MA_Cnty_SA_{y}_{m}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing service area file: {path}")
    df = read_service_area(path)
    df["month"] = int(m)
    df["year"] = y
    return df


def read_penetration(path: str) -> pd.DataFrame:
    """Read penetration CSV file."""
    col_names = [
        "state", "county", "fips_state", "fips_cnty", "fips",
        "ssa_state", "ssa_cnty", "ssa", "eligibles", "enrolled", "penetration",
    ]
    df = pd.read_csv(
        path,
        skiprows=1,
        names=col_names,
        na_values=["", "NA", "*", "-", "--"],
        dtype={
            "state": str,
            "county": str,
            "fips_state": "Int64",
            "fips_cnty": "Int64",
            "fips": float,
            "ssa_state": "Int64",
            "ssa_cnty": "Int64",
            "ssa": float,
            "eligibles": str,
            "enrolled": str,
            "penetration": str,
        },
    )

    # Parse numeric columns (handles commas and %)
    for col in ["eligibles", "enrolled", "penetration"]:
        # df[col] may have missing values; convert to string safely
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce",
        )

    return df


def load_month_pen(m: str, y: int) -> pd.DataFrame:
    """Load one month of penetration data."""
    path = MA_DATA_DIR / "ma" / "penetration" / "Extracted Data" / f"State_County_Penetration_MA_{y}_{m}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing penetration file: {path}")
    df = read_penetration(path)
    df["month"] = int(m)
    df["year"] = y
    return df


def mapd_clean_merge(ma_data: pd.DataFrame, mapd_data: pd.DataFrame, y: int) -> pd.DataFrame:
    """Clean and merge MA and MA-PD landscape data."""
    # Tidy MA-only data
    ma_data = ma_data[["contractid", "planid", "state", "county", "premium"]].copy()
    ma_data = ma_data.sort_values(["contractid", "planid", "state", "county"])
    ma_data["premium"] = ma_data.groupby(["contractid", "planid", "state", "county"])["premium"].ffill()
    ma_data = ma_data.drop_duplicates(subset=["contractid", "planid", "state", "county"], keep="first")

    # Tidy MA-PD data
    mapd_data = mapd_data[
        [
            "contractid", "planid", "state", "county",
            "premium_partc", "premium_partd_basic", "premium_partd_supp",
            "premium_partd_total", "partd_deductible",
        ]
    ].copy()
    mapd_data["planid"] = pd.to_numeric(mapd_data["planid"], errors="coerce")

    mapd_data = mapd_data.sort_values(["contractid", "planid", "state", "county"])
    fill_cols = [
        "premium_partc", "premium_partd_basic", "premium_partd_supp",
        "premium_partd_total", "partd_deductible",
    ]
    mapd_data[fill_cols] = mapd_data.groupby(["contractid", "planid", "state", "county"])[fill_cols].ffill()
    mapd_data = mapd_data.drop_duplicates(subset=["contractid", "planid", "state", "county"], keep="first")

    # Merge Part D info to Part C info
    plan_premiums = ma_data.merge(mapd_data, on=["contractid", "planid", "state", "county"], how="outer")
    plan_premiums["year"] = y
    return plan_premiums