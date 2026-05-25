# data_processing.py

import pandas as pd
from dataloading import load_data

LEAKAGE_COLS = [
    'patient_id',
    'tb_type',
    'treatment_started',
    'treatment_category',
    'treatment_outcome',
    'tb_probability_score',
    'died',
    'tb_status',
    'cd4_count'
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs cleaning + leakage removal + basic preprocessing.
    """

    df_clean = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    # Handle missing values (simple medical-safe defaults)
    if "symptom_duration_weeks" in df_clean.columns:
        df_clean["symptom_duration_weeks"] = df_clean["symptom_duration_weeks"].fillna(0)

    if "smear_status" in df_clean.columns:
        df_clean["smear_status"] = df_clean["smear_status"].fillna("Negative")

    # Fill boolean-like medical indicators
    cols_to_fill_false = [
        'culture_confirmed',
        'cavitary_disease',
        'mdr_tb',
        'xdr_tb',
        'xray_abnormal'
    ]

    for col in cols_to_fill_false:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(False)

    print("Data cleaned successfully.")
    print(f"Shape after cleaning: {df_clean.shape}")

    return df_clean


dataset = load_data()

print(clean_data(dataset))