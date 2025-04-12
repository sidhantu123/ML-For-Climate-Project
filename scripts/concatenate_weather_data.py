import pandas as pd
import os
import glob

def combine_weather_data():
    # Update path to weather_data directory
    base_dir = "/Users/sidhantumbrajkar/coding_projects/ML-For-Climate-Project"
    weather_dir = os.path.join(base_dir, "scripts", "weather_data")
    
    # Check if directory exists
    if not os.path.exists(weather_dir):
        print(f"Error: {weather_dir} directory not found!")
        return
    
    # List all CSV files in the directory
    csv_files = glob.glob(os.path.join(weather_dir, "*_weather.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {weather_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to combine")
    
    # List to store dataframes
    dfs = []
    
    # Read each CSV file
    for file in csv_files:
        try:
            # Extract city name from filename
            city = os.path.basename(file).replace('_weather.csv', '')
            
            # Read the CSV
            df = pd.read_csv(file)
            
            # Add city column if not present
            if 'city' not in df.columns:
                df['city'] = city
            
            dfs.append(df)
            print(f"Processed: {file}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not dfs:
        print("No data frames to combine!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by city and date if date column exists
    if 'datetime' in combined_df.columns:
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
        combined_df = combined_df.sort_values(['city', 'datetime'])
    
    # Save combined file in the root directory
    output_file = "master_weather_data.csv"
    combined_df.to_csv(output_file, index=False)
    
    print("\nSummary:")
    print(f"Total files processed: {len(dfs)}")
    print(f"Total rows in combined file: {len(combined_df)}")
    print(f"Columns in combined file: {', '.join(combined_df.columns)}")
    print(f"Data saved to: {output_file}")
    
    # Print data for each city
    print("\nRows per city:")
    print(combined_df['city'].value_counts())

if __name__ == "__main__":
    combine_weather_data()