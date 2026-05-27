import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

def automate_preprocessing(input_path, output_path):
    """
    Fungsi otomatisasi untuk memproses data mentah (raw) 
    menjadi data yang siap dilatih oleh model Machine Learning.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File mentah tidak ditemukan di: {input_path}")
    df = pd.read_csv(input_path)
    if "User_ID" in df.columns:
        df.drop("User_ID", axis=1, inplace=True)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "male" else 0)
    df.dropna(inplace=True)
    
    features_to_transform = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Calories']
    columns_in_data = [col for col in features_to_transform if col in df.columns]
    if columns_in_data:
        pt = PowerTransformer(method='box-cox', standardize=True)
        df[columns_in_data] = pt.fit_transform(df[columns_in_data])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Selesai! Data steril berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "calories.csv"
    PROCESSED_DATA_PATH = "preprocessing/calories_processed.csv"
    automate_preprocessing(RAW_DATA_PATH, PROCESSED_DATA_PATH)