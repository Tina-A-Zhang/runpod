import pandas as pd
import numpy as np
import os

def silver_layer_transformation_chunked(input_file_path, output_file_path, chunk_size=50000):
    """
    Cleans and prepares data in chunks, then saves the result to a new Parquet file.
    
    Args:
        input_file_path (str): The path to the Bronze layer Parquet file.
        output_file_path (str): The path where the output Parquet file will be saved.
        chunk_size (int): The number of rows to process at a time.
    """
    # Use a flag to write the file header only once
    header_written = False

    # Check if the output directory exists, if not, create it
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Read the Bronze layer Parquet file in chunks
    for chunk in pd.read_parquet(input_file_path, engine='pyarrow', columns=None):
        
        # --- 1) Normalize/screen loan_status to final outcomes only ---
        final_status = ["Fully Paid", "Charged Off", "Default"]
        chunk['loan_status'] = chunk['loan_status'].replace({
            "Does not meet the credit policy. Status:Fully Paid": "Fully Paid",
            "Does not meet the credit policy. Status:Charged Off": "Charged Off",
        })
        chunk = chunk[chunk['loan_status'].isin(final_status)]

        # --- 2) Binary target ---
        chunk['is_bad_loan'] = chunk['loan_status'].isin(["Charged Off", "Default"]).astype(int)

        # --- 3) Drop leakage/low-value columns ---
        drop_candidates = [
            "id", "member_id", "url", "desc", "policy_code", "emp_title", "title", "recoveries",
            "collection_recovery_fee", "last_pymnt_d", "last_credit_pull_d", "total_rec_prncp",
            "total_rec_int", "total_rec_late_fee", "last_fico_range_high", "last_fico_range_low",
            "total_pymnt", "total_pymnt_inv", "out_prncp", "out_prncp_inv", "next_pymnt_d",
            "last_pymnt_amnt", "hardship_flag", "hardship_type", "hardship_reason",
            "hardship_status", "hardship_amount", "hardship_start_date",
            "hardship_end_date", "debt_settlement_flag", "debt_settlement_flag_date",
            "settlement_status", "settlement_date", "settlement_amount",
            "loan_status",  # drop after creating is_bad_loan
        ]
        
        chunk.drop(columns=[c for c in drop_candidates if c in chunk.columns], inplace=True)

        # --- 4) Type cleaning ---
        if 'int_rate' in chunk.columns:
            chunk['int_rate'] = chunk['int_rate'].str.replace('%', '').astype(float)
        
        if 'revol_util' in chunk.columns:
            chunk['revol_util'] = chunk['revol_util'].str.replace('%', '').astype(float)

        # term: "36 months" -> 36
        if 'term' in chunk.columns:
            chunk['term_months'] = chunk['term'].str.extract(r'(\d+)').astype(int)
            chunk.drop('term', axis=1, inplace=True)

        # emp_length -> months, None if unknown
        if 'emp_length' in chunk.columns:
            chunk['emp_length_months'] = chunk['emp_length'].str.replace(r'[^0-9]+', '', regex=True).astype(float) * 12
            chunk.loc[chunk['emp_length'].str.contains('< 1', na=False), 'emp_length_months'] = 0
            chunk.loc[chunk['emp_length'].str.contains('10\+', na=False), 'emp_length_months'] = 120
            chunk.drop('emp_length', axis=1, inplace=True)

        # Ordered encoding for sub_grade: A1..G5 -> 0..34
        if 'sub_grade' in chunk.columns:
            # Create a lookup table for sub_grades
            sub_grade_map = {f'{grade}{num}': i for i, grade in enumerate('ABCDEFG') for num in range(1, 6)}
            chunk['sub_grade_ord'] = chunk['sub_grade'].map(sub_grade_map)

        # Home ownership normalization (allow-list + case-normalize)
        if 'home_ownership' in chunk.columns:
            valid_ownership = ["MORTGAGE", "OWN", "RENT"]
            chunk['home_ownership'] = chunk['home_ownership'].str.upper()
            chunk['home_ownership_norm'] = np.where(chunk['home_ownership'].isin(valid_ownership), chunk['home_ownership'], 'OTHER')

        # --- 5) Dates & temporal helpers (keep issue_date) ---
        if 'issue_d' in chunk.columns:
            chunk['issue_date'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y')
            chunk['issue_year'] = chunk['issue_date'].dt.year
            chunk['issue_month'] = chunk['issue_date'].dt.month
        
        if 'earliest_cr_line' in chunk.columns:
            chunk['earliest_credit_line_date'] = pd.to_datetime(chunk['earliest_cr_line'], format='%b-%Y')

        if 'issue_date' in chunk.columns and 'earliest_credit_line_date' in chunk.columns:
            chunk['credit_age_months'] = ((chunk['issue_date'] - chunk['earliest_credit_line_date']) / np.timedelta64(1, 'M')).astype(int).clip(lower=0)

        # --- 6) Basic sanity filters ---
        chunk = chunk[chunk['annual_inc'].notna() & (chunk['annual_inc'] > 0)]
        chunk = chunk[chunk['dti'].notna() & (chunk['dti'] >= 0) & (chunk['dti'] <= 65)]
        chunk = chunk[chunk['revol_util'].notna() & (chunk['revol_util'] >= 0) & (chunk['revol_util'] <= 150)]

        # --- 7) Keep & dedupe on the new key ---
        chunk = chunk[chunk['loan_application_id'].notna()]
        chunk.drop_duplicates(subset=['loan_application_id'], inplace=True)

        # Write the chunk to a Parquet file
        if not header_written:
            chunk.to_parquet(output_file_path, index=False)
            header_written = True
        else:
            chunk.to_parquet(output_file_path, index=False, mode='a')
    
    print(f"Silver layer processing complete. Output saved to {output_file_path}")

if __name__ == '__main__':
    # This block allows you to run the script directly for testing.
    bronze_data_file = 'data/bronze_output.parquet'
    silver_output_file = 'data/silver_output.parquet'
    silver_layer_transformation_chunked(bronze_data_file, silver_output_file)