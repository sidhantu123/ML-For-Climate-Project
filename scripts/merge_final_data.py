import pandas as pd
import os

def merge_air_quality_with_weather_ozone():
    # Set up paths
    base_dir = "/Users/sidhantumbrajkar/coding_projects/ML-For-Climate-Project"
    
    # Read the files
    print("Reading data files...")
    weather_ozone_df = pd.read_csv(os.path.join(base_dir, "weather_and_ozone_data.csv"))
    city_day_df = pd.read_csv(os.path.join(base_dir, "city_day.csv"))
    
    # Process weather_ozone_df
    print("Processing weather and ozone data...")
    # Extract city name from the complex city string (e.g., "Ahmedabad, GJ, India" -> "Ahmedabad")
    weather_ozone_df['city'] = weather_ozone_df['city'].str.split(',').str[0].str.strip()
    weather_ozone_df['Date'] = pd.to_datetime(weather_ozone_df['datetime']).dt.date
    
    # Process city_day_df
    print("Processing city day data...")
    city_day_df['Date'] = pd.to_datetime(city_day_df['Date']).dt.date
    
    # Standardize city names (convert to lowercase for matching)
    weather_ozone_df['city'] = weather_ozone_df['city'].str.lower()
    city_day_df['City'] = city_day_df['City'].str.lower()
    
    # Merge the dataframes
    print("Merging datasets...")
    merged_df = pd.merge(
        weather_ozone_df,
        city_day_df,
        left_on=['city', 'Date'],
        right_on=['City', 'Date'],
        how='outer'
    )
    
    # Clean up duplicate columns if they exist
    if 'City' in merged_df.columns:
        merged_df.drop('City', axis=1, inplace=True)
    
    # Save the final merged dataset
    output_file = os.path.join(base_dir, "final_merged_data.csv")
    print(f"Saving merged data to {output_file}")
    merged_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\nMerge Summary:")
    print(f"Weather and Ozone records: {len(weather_ozone_df)}")
    print(f"City Day records: {len(city_day_df)}")
    print(f"Final merged records: {len(merged_df)}")
    
    # Check for missing values in key columns
    print("\nMissing values in key columns:")
    columns_to_check = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'AQI', 
                       'temp', 'humidity', 'windspeed',
                       'TropoO3', 'OMIO3', 'StratoO3']
    # Only check columns that exist
    existing_columns = [col for col in columns_to_check if col in merged_df.columns]
    missing_values = merged_df[existing_columns].isnull().sum()
    print(missing_values)
    
    # Show sample of unique cities in final dataset
    print("\nUnique cities in merged dataset:")
    print(merged_df['city'].nunique())
    print(merged_df['city'].unique())

if __name__ == "__main__":
    merge_air_quality_with_weather_ozone()