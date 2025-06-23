import pandas as pd
import os
from multiprocessing import Pool

def process_single_file(args):
    """Worker function to process a single CSV file"""
    csv_file, folder_path = args
    # Get study_id from filename
    study_id = csv_file.replace('_freqAltas.csv', '')
    
    # Read and process CSV
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    df.set_index('Gene', inplace=True)

    # Select and rename frequency column
    return df[['Frequency (%)']].rename(columns={'Frequency (%)': study_id})


def load_and_combine_csv_files(folder_path, condition_file, study_name):
    """Load and combine CSV files in parallel"""
    # Get list of CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('_freqAltas.csv')]
    
    # Process files in parallel
    with Pool() as pool:
        # Create args list for worker function
        args = [(f, folder_path) for f in csv_files]
        # Map process_single_file across all files
        dataframes = pool.map(process_single_file, args)
    
    # Combine all DataFrames on Gene index
    master_df = dataframes[0]
    for df in dataframes[1:]:
        master_df = master_df.join(df, how='outer')
    
    # Process condition file (depression/other disease)
    condition_df = pd.read_csv(condition_file)
    
    # Ensure Gene column is the first column and set as index
    if 'Gene' in condition_df.columns:
        condition_df = condition_df.set_index('Gene')
    else:
        # If no 'Gene' column, assume first column contains gene names
        condition_df = condition_df.set_index(condition_df.columns[0])
    
    # Get the frequency column (assume it's the first remaining column after setting index)
    freq_column = condition_df.columns[0]
    condition_series = condition_df[freq_column].rename(study_name)
    
    # Add condition data to master DataFrame
    master_df = master_df.join(condition_series, how='outer')
    
    # Save final result
    # output_path = 'Results/ClustersCSV/master_df.csv'
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # master_df.to_csv(output_path)
    
    return master_df