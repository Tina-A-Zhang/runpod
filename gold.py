import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib # to save the transformers for later use

def gold_layer_transformation_chunked(input_file_path, output_file_path, chunk_size=50000):
    """
    Processes the Silver layer data in chunks, performs feature engineering,
    and saves the final features to a new Parquet file.
    
    Args:
        input_file_path (str): The path to the Silver layer Parquet file.
        output_file_path (str): The path where the output Parquet file will be saved.
        chunk_size (int): The number of rows to process at a time.
    """
    
    # Define a list of features to process
    categorical_features = ["purpose", "home_ownership_norm", "verification_status"]
    numeric_features = [
        "loan_amnt", "installment", "annual_inc", "fico_range_high", 
        "fico_range_low", "int_rate", "dti", "open_acc", "pub_rec", 
        "revol_bal", "revol_util", "total_acc", "sub_grade_ord", 
        "term_months", "emp_length_months", "credit_age_months", 
        "issue_year", "issue_month",
    ]

    # Initialize transformers
    imputer = SimpleImputer(strategy='median')
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    header_written = False
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # The first pass is for fitting the transformers on a sample of data
    print("Fitting transformers on first chunk...")
    for chunk in pd.read_parquet(input_file_path, engine='pyarrow', chunksize=chunk_size):
        # Fit Imputer and Encoder on the first chunk
        numeric_chunk = chunk[numeric_features]
        imputer.fit(numeric_chunk)
        
        categorical_chunk = chunk[categorical_features]
        encoder.fit(categorical_chunk)
        break  # We only need the first chunk to fit

    # Save the fitted transformers so they can be reused for inference
    joblib.dump(imputer, 'imputer.joblib')
    joblib.dump(encoder, 'encoder.joblib')
    
    # Now, process the full dataset chunk by chunk
    print("Processing entire dataset...")
    for chunk in pd.read_parquet(input_file_path, engine='pyarrow', chunksize=chunk_size):
        
        # --- Label hygiene ---
        if "label" in chunk.columns:
            chunk['label'] = chunk['label'].astype('float')
        elif "is_bad_loan" in chunk.columns:
            chunk['label'] = chunk['is_bad_loan'].astype('float')
        else:
            chunk['label'] = np.nan

        # --- Impute Numeric features ---
        chunk[numeric_features] = imputer.transform(chunk[numeric_features])

        # --- One-Hot Encode Categorical features ---
        encoded_features = encoder.transform(chunk[categorical_features])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
        
        # Reset index to allow merging
        chunk = chunk.reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)

        # Concatenate imputed numerics and encoded categoricals
        final_features = pd.concat([chunk[numeric_features], encoded_df], axis=1)

        # Add `loan_application_id` and `label` back
        final_chunk = pd.concat([chunk[['loan_application_id', 'label']], final_features], axis=1)
        
        # Write the chunk to a Parquet file
        if not header_written:
            final_chunk.to_parquet(output_file_path, index=False)
            header_written = True
        else:
            final_chunk.to_parquet(output_file_path, index=False, mode='a')
            
    print(f"Gold layer processing complete. Output saved to {output_file_path}")

if __name__ == '__main__':
    silver_data_file = 'data/silver_output.parquet'
    gold_output_file = 'data/gold_output.parquet'
    gold_layer_transformation_chunked(silver_data_file, gold_output_file)