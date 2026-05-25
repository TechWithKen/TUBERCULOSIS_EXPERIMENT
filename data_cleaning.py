# data_cleaning.py

import pandas as pd

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

def create_label(x) -> str:
    """
    Performs target label creation before removing the column necessary.
    """

    if x > 0.7:
        return "High"
    elif x > 0.4:
        return "Medium"
    else:
        return "Low"
    

def apply_label(df: pd.DataFrame) -> pd.DataFrame:

    """
    Applying the created standards of the dataframe.
    """
    
    df['tb_risk'] = df['tb_probability_score'].apply(create_label)

    return df


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

