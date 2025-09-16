# bronze.py
# (The code is the same as the previous response)
import pandas as pd
import hashlib
import os

def bronze_layer_transformation_chunked(input_file_path, output_file_path, chunk_size=50000):
    """
    Processes a large CSV file in chunks, adding a unique hash ID to each record
    and saves the output to a Parquet file.
    
    Args:
        input_file_path (str): The path to the raw data CSV file.
        output_file_path (str): The path where the output Parquet file will be saved.
        chunk_size (int): The number of rows to process at a time.
    """
    # Define key columns to create a stable, unique ID.
    STABLE_ID_COLS = [
        "loan_amnt", "funded_amnt", "int_rate", "installment", "grade",
        "home_ownership", "annual_inc", "dti", "purpose", "addr_state",
    ]

    # Use a flag to write the file header only once
    header_written = False
    
    # Check if the output directory exists, if not, create it
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Read the CSV in chunks
    for chunk in pd.read_csv(input_file_path, chunksize=chunk_size):
        # Create a combined key by concatenating strings from stable columns
        chunk['combined_key'] = chunk[STABLE_ID_COLS].astype(str).sum(axis=1)

        # Apply a hash function to the new combined_key column to create a unique ID.
        chunk['loan_application_id'] = chunk['combined_key'].apply(
            lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()
        )

        # Drop the temporary combined_key column.
        chunk.drop('combined_key', axis=1, inplace=True)

        # Write the chunk to a Parquet file
        if not header_written:
            chunk.to_parquet(output_file_path, index=False)
            header_written = True
        else:
            chunk.to_parquet(output_file_path, index=False, mode='a')
            
    print(f"Bronze layer processing complete. Output saved to {output_file_path}")

if __name__ == '__main__':
    raw_data_file = 'data/raw_data.csv'
    bronze_output_file = 'data/bronze_output.parquet'
    bronze_layer_transformation_chunked(raw_data_file, bronze_output_file)