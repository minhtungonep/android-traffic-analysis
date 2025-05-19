import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to clean the Android traffic dataset
def clean_android_traffic_data(file_path):
    """
    Clean the Android traffic dataset by:
    1. Fixing column names
    2. Standardising type values
    3. Converting non-numeric values in numeric columns to NaN
    4. Handling missing values
    5. Removing rows with critical missing data
    
    Parameters:
    file_path (str): Path to the raw CSV file
    
    Returns:
    pd.DataFrame: Cleaned dataset
    dict: Data cleaning report with information about the changes made
    """
    
    # Try different encoding standards, because I ran into an issue with default encoding
    encodings = ['utf-8', 'cp1252', 'latin1', 'ISO-8859-1', 'utf-16']
    df = None
    encoding_errors = {}
    
    for encoding in encodings:
        try:
            print(f"Attempting to load data with {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded data with {encoding} encoding")
            break
        except UnicodeDecodeError as e:
            error_msg = str(e)
            encoding_errors[encoding] = error_msg
            print(f"Failed to load with {encoding} encoding: {error_msg}")
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            encoding_errors[encoding] = f"{error_type}: {error_msg}"
            print(f"Error with {encoding} encoding: {error_type}: {error_msg}")
    
    # If all encoding attempts failed, provide comprehensive error report
    if df is None:
        error_report = "\n".join([f"{enc}: {err}" for enc, err in encoding_errors.items()])
        raise ValueError(f"Failed to load the file with any of the attempted encodings. Error details:\n{error_report}\n"
                         f"Consider checking file integrity or manually specifying the correct encoding.")
    
    # Create a report dictionary to track changes
    report = {
        'encoding_used': encoding,
        'encoding_attempts': list(encoding_errors.keys()) + [encoding],
        'original_shape': df.shape,
        'data_quality_issues': [],
        'columns_renamed': {},
        'non_numeric_values': {},
        'standardised_types': {},
        'rows_removed': 0
    }
    
    # 1. Fix column names
    print("Fixing column names...")
    # Check for and rename misspelled 'vulume_bytes' column
    if 'vulume_bytes' in df.columns:
        df.rename(columns={'vulume_bytes': 'volume_bytes'}, inplace=True)
        report['columns_renamed']['vulume_bytes'] = 'volume_bytes'
    
    # Check for and rename columns with trailing spaces
    for col in df.columns:
        if col.strip() != col:
            df.rename(columns={col: col.strip()}, inplace=True)
            report['columns_renamed'][col] = col.strip()
    
    # Identify and handle duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated(keep=False)].tolist()
    if duplicate_cols:
        report['data_quality_issues'].append(f"Duplicate columns found: {duplicate_cols}")
        # If 'source_app_packets' is duplicated, rename the second occurrence
        if 'source_app_packets' in duplicate_cols:
            # Find the position of duplicates
            cols = df.columns.tolist()
            duplicate_indices = [i for i, col in enumerate(cols) if col == 'source_app_packets']
            if len(duplicate_indices) > 1:
                # Rename the second occurrence
                cols[duplicate_indices[1]] = 'source_app_packets_duplicate'
                df.columns = cols
                report['columns_renamed'][f'source_app_packets at index {duplicate_indices[1]}'] = 'source_app_packets_duplicate'
    
    # 2. Standardise type values
    print("Standardising type values...")
    # Get unique values in the 'type' column before standardisation
    original_types = df['type'].unique().tolist()
    report['standardised_types']['original_values'] = original_types
    
    # Create a function to standardise type values
    def standardise_type(type_val):
        if isinstance(type_val, str):
            if 'benign' in type_val.lower():
                return 'benign'
            elif type_val == 'malicious':
                return 'malicious'
            else:
                return 'unknown'  # For unrecognised values
        return 'unknown'  # For non-string values
    
    df['type'] = df['type'].apply(standardise_type)
    report['standardised_types']['standardised_values'] = df['type'].unique().tolist()
    
    # 3. Convert non-numeric values in numeric columns to NaN
    print("Handling non-numeric values in numeric columns...")
    # List of columns that should contain numeric data
    numeric_columns = [
        'tcp_packets', 'dist_port_tcp', 'external_ips', 'volume_bytes', 
        'udp_packets', 'tcp_urg_packet', 'source_app_packets', 
        'remote_app_packets', 'source_app_bytes', 'remote_app_bytes',
        'dns_query_times'
    ]
    
    # If renamed 'vulume_bytes' to 'volume_bytes', update the list
    if 'vulume_bytes' in report['columns_renamed']:
        numeric_columns = [col if col != 'vulume_bytes' else 'volume_bytes' for col in numeric_columns]
    
    # Add 'source_app_packets_duplicate' if it exists
    if 'source_app_packets_duplicate' in df.columns:
        numeric_columns.append('source_app_packets_duplicate')
    
    # Check for and record non-numeric values before conversion
    for col in numeric_columns:
        if col in df.columns:
            try:
                # First, check for and record non-numeric values
                temp_col = df[col].astype(str).replace('NA', np.nan)
                non_numeric_mask = ~pd.to_numeric(temp_col, errors='coerce').notna() & temp_col.notna()
                non_numeric_values = temp_col[non_numeric_mask]
                
                if len(non_numeric_values) > 0:
                    report['non_numeric_values'][col] = {
                        'count': len(non_numeric_values),
                        'examples': non_numeric_values.unique().tolist()[:5],  # Show up to 5 examples
                        'unique_count': len(non_numeric_values.unique())
                    }
                    print(f"Warning: Found {len(non_numeric_values)} non-numeric values in column '{col}'")
                
                # Then convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            except Exception as e:
                error_type = type(e).__name__
                print(f"Error processing column '{col}': {error_type}: {str(e)}")
                report['data_quality_issues'].append(f"Error in column '{col}': {error_type}: {str(e)}")
                # Attempt to salvage the column by force-converting to numeric
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).replace(['NA', 'null', ''], np.nan), errors='coerce')
                    print(f"  - Salvaged column '{col}' by force-converting to numeric with NaN for non-numeric values")
                except:
                    print(f"  - Could not salvage column '{col}'. Setting all values to NaN")
                    df[col] = np.nan
    
    # Convert columns to numeric, replacing non-numeric values with NaN
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Handle missing values
    print("Handling missing values...")
    # Replace "NA" strings with NaN
    df.replace("NA", np.nan, inplace=True)
    
    # Record missing values
    missing_values = df.isnull().sum()
    report['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    # 5. Remove rows with critical missing data
    print("Removing rows with critical missing data...")
    # For hypothesis 1, I need dns_query_times and tcp_packets
    # For hypothesis 2, I need volume_bytes
    # I also need valid type values
    
    # Get initial count
    initial_count = len(df)
    
    # Filter out rows with missing critical data
    critical_columns = ['dns_query_times', 'tcp_packets', 'volume_bytes']
    df_clean = df.dropna(subset=critical_columns)
    
    # Filter out rows with unknown type
    df_clean = df_clean[df_clean['type'].isin(['benign', 'malicious'])]
    
    # Calculate removed rows
    report['rows_removed'] = initial_count - len(df_clean)
    report['cleaned_shape'] = df_clean.shape
    
    print(f"Data cleaning completed: {report['rows_removed']} rows removed")
    print(f"Original shape: {report['original_shape']}, Cleaned shape: {report['cleaned_shape']}")
    
    return df_clean, report

# Function to save the cleaned data and generate a report
def save_cleaned_data(df_clean, report, output_csv_path, report_path=None):
    """
    Save the cleaned data to a CSV file and generate a cleaning report
    
    Parameters:
    df_clean (pd.DataFrame): Cleaned dataset
    report (dict): Data cleaning report
    output_csv_path (str): Path to save the cleaned CSV file
    report_path (str, optional): Path to save the cleaning report text file
    """
    # Save cleaned data
    df_clean.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to {output_csv_path}")
    
    # Generate and save report if path is provided
    if report_path:
        with open(report_path, 'w') as f:
            f.write("Android Traffic Data Cleaning Report\n")
            f.write("===================================\n\n")
            
            f.write(f"Encoding used: {report['encoding_used']}\n\n")
            f.write(f"Original data shape: {report['original_shape'][0]} rows, {report['original_shape'][1]} columns\n")
            f.write(f"Cleaned data shape: {report['cleaned_shape'][0]} rows, {report['cleaned_shape'][1]} columns\n")
            f.write(f"Rows removed: {report['rows_removed']}\n\n")
            
            f.write("Columns Renamed:\n")
            for old_name, new_name in report['columns_renamed'].items():
                f.write(f"  - '{old_name}' -> '{new_name}'\n")
            f.write("\n")
            
            f.write("Type Standardisation:\n")
            f.write(f"  - Original values: {report['standardised_types']['original_values']}\n")
            f.write(f"  - Standardised values: {report['standardised_types']['standardised_values']}\n")
            f.write("\n")
            
            f.write("Non-numeric Values Found:\n")
            for col, info in report['non_numeric_values'].items():
                f.write(f"  - Column '{col}': {info['count']} non-numeric values\n")
                f.write(f"    Examples: {info['examples']}\n")
            f.write("\n")
            
            f.write("Missing Values by Column:\n")
            for col, count in report['missing_values'].items():
                f.write(f"  - {col}: {count} missing values\n")
            
        print(f"Cleaning report saved to {report_path}")

# Function to visualise the data cleaning process
def visualise_cleaning_process(df_clean, report, output_dir):
    """
    Create visualisations to illustrate the data cleaning process
    
    Parameters:
    df_clean (pd.DataFrame): Cleaned dataset
    report (dict): Data cleaning report
    output_dir (str): Directory to save the visualisations
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Bar chart of missing values before cleaning
    plt.figure(figsize=(12, 6))
    missing_cols = {k: v for k, v in report['missing_values'].items() if v > 0}
    plt.bar(missing_cols.keys(), missing_cols.values())
    plt.title('Missing Values by Column Before Cleaning')
    plt.xlabel('Column')
    plt.ylabel('Count of Missing Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values.png'))
    plt.close()
    
    # 2. Pie chart of traffic types after standardisation
    plt.figure(figsize=(8, 8))
    type_counts = df_clean['type'].value_counts()
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Traffic Types After Cleaning')
    plt.savefig(os.path.join(output_dir, 'traffic_types.png'))
    plt.close()
    
    # 3. Before/After row counts (to show impact of cleaning)
    plt.figure(figsize=(10, 6))
    stages = ['Original Data', 'After Cleaning']
    counts = [report['original_shape'][0], report['cleaned_shape'][0]]
    plt.bar(stages, counts, color=['#aaaaaa', '#4CAF50'])
    plt.title('Row Count Before and After Cleaning')
    plt.ylabel('Number of Rows')
    for i, count in enumerate(counts):
        plt.text(i, count + 50, str(count), ha='center')
    plt.savefig(os.path.join(output_dir, 'row_counts.png'))
    plt.close()
    
    print(f"Visualisations saved to {output_dir}")

# Main function to run the data cleaning process
def main():
    # Set file paths
    input_file = 'data/android_traffic-not clean.csv'
    output_file = 'data/android_traffic_clean.csv'
    report_file = 'cleaning_visualisations/cleaning_report.txt'
    viz_dir = 'cleaning_visualisations'
    
    # Clean the data
    df_clean, report = clean_android_traffic_data(input_file)
    
    # Save results
    save_cleaned_data(df_clean, report, output_file, report_file)
    
    # Create visualisations
    visualise_cleaning_process(df_clean, report, viz_dir)
    
    print("Data cleaning process completed successfully!")
    return df_clean

# Run the script
if __name__ == "__main__":
    df_clean = main()