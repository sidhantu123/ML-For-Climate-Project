import pandas as pd
import os


# Read the city_day.csv file
base_dir = "/Users/sidhantumbrajkar/coding_projects/ML-For-Climate-Project"
city_day_df = pd.read_csv(os.path.join(base_dir, "city_day.csv"))
city_day_df["City"] = city_day_df[
    "City"
].str.lower()  # Standardize city names to lowercase

# Read the weather_and_ozone_data.csv file
weather_df = pd.read_csv(os.path.join(base_dir, "weather_and_ozone_data.csv"))
# Clean up city names by removing state and country info
weather_df["clean_name"] = (
    weather_df["name"].str.split(",").str[0].str.strip().str.lower()
)
weather_df.loc[
    weather_df["clean_name"] == "brajarajnagar", "clean_name"
] = "brajrajnagar"

# Convert datetime columns to same format for merging
weather_df["datetime"] = pd.to_datetime(weather_df["datetime"]).dt.date
city_day_df["Date"] = pd.to_datetime(city_day_df["Date"]).dt.date

# Merge the dataframes on city and date
merged_df = pd.merge(
    city_day_df,
    weather_df,
    left_on=["City", "Date"],
    right_on=["clean_name", "datetime"],
    how="inner",  # only keep rows that match in both datasets
)

# Drop duplicate columns (clean_name and datetime since we already have City and Date)
merged_df = merged_df.drop(["clean_name", "datetime", "name"], axis=1)

# Save the merged dataset
output_path = os.path.join(base_dir, "merged_weather_air_quality.csv")
merged_df.to_csv(output_path, index=False)

# Print some information about the merge
print(f"Original city_day.csv rows: {len(city_day_df)}")
print(f"Original weather_and_ozone_data.csv rows: {len(weather_df)}")
print(f"Merged dataset rows: {len(merged_df)}")
print(f"\nMerged data saved to: {output_path}")

# Display the first few columns of the merged dataset to verify
print("\nFirst few columns of the merged dataset:")
print(merged_df.columns.tolist()[:10])
