import pandas as pd
import os

def load_hospitals():
    """
    Loads the hospital dataset and cleans up the column names.
    """
    file_path = os.path.join(os.path.dirname(__file__), "../datasets/Hospital_Dataset.xlsx")
    hospitals_df = pd.read_excel(file_path)
    # Clean up column names by stripping whitespace
    hospitals_df.columns = hospitals_df.columns.str.strip()
    return hospitals_df

def load_diseases():
    """
    Loads the disease-treatment dataset and cleans up the column names.
    """
    file_path = os.path.join(os.path.dirname(__file__), "../datasets/disease_treatment_dataset.xlsx")
    diseases_df = pd.read_excel(file_path)
    # Clean up column names by stripping whitespace
    diseases_df.columns = diseases_df.columns.str.strip()
    return diseases_df

def load_claims_history():
    """
    Loads the claims history dataset and cleans up the column names.
    """
    file_path = os.path.join(os.path.dirname(__file__), "../datasets/Claims_Fraud_Dataset_with_Reasons.xlsx")
    claims_df = pd.read_excel(file_path)
    # Clean up column names by stripping whitespace
    claims_df.columns = claims_df.columns.str.strip()
    return claims_df
