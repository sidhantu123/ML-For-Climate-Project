import pandas as pd
import os

def merge_weather_and_ozone():
    # Absolute paths
    base_dir = "/Users/sidhantumbrajkar/coding_projects/ML-For-Climate-Project"
    scripts_dir = os.path.join(base_dir, "scripts")
    
    # Read the files
    print("Reading data files...")
    weather_df = pd.read_csv(os.path.join(scripts_dir, "master_weather_data.csv"))
    ozone_df = pd.read_csv(os.path.join(scripts_dir, "city_ozone_data.csv"))
    
    # Ensure date columns are in the same format
    print("Processing dates...")
    weather_df['Date'] = pd.to_datetime(weather_df['datetime']).dt.date
    ozone_df['Date'] = pd.to_datetime(ozone_df['Date']).dt.date
    
    # Ensure city names match (convert to lowercase for consistency)
    weather_df['City'] = weather_df['city'].str.lower()
    ozone_df['City'] = ozone_df['City'].str.lower()
    
    # Merge the dataframes on City and Date
    print("Merging datasets...")
    merged_df = pd.merge(
        weather_df,
        ozone_df[['City', 'Date', 'TropoO3', 'OMIO3', 'StratoO3']],
        on=['City', 'Date'],
        how='left'
    )
    
    # Clean up date columns
    merged_df.drop('Date', axis=1, inplace=True)
    
    # Save the merged dataset
    output_file = os.path.join(base_dir, "weather_and_ozone_data.csv")
    print(f"Saving merged data to {output_file}")
    merged_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nMerge Summary:")
    print(f"Weather records: {len(weather_df)}")
    print(f"Ozone records: {len(ozone_df)}")
    print(f"Merged records: {len(merged_df)}")
    print("\nSample of columns in merged file:")
    print(merged_df.columns.tolist())
    
    # Check for any missing values in ozone columns
    print("\nMissing values in ozone columns:")
    print(merged_df[['TropoO3', 'OMIO3', 'StratoO3']].isnull().sum())

if __name__ == "__main__":
    merge_weather_and_ozone()
