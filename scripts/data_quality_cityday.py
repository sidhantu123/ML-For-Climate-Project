import pandas as pd
import os

# Read the city_day dataset
base_dir = "/Users/sidhantumbrajkar/coding_projects/ML-For-Climate-Project"
city_day_df = pd.read_csv(os.path.join(base_dir, "city_day.csv"))

# Group by city and count missing AQI values
aqi_missing = (
    city_day_df.groupby("City")["AQI"].apply(lambda x: x.isna().sum()).reset_index()
)
total_records = city_day_df.groupby("City").size().reset_index(name="total_records")

# Merge the counts with total records
aqi_stats = pd.merge(aqi_missing, total_records, on="City")

# Calculate percentage of missing values
aqi_stats["missing_percentage"] = (
    aqi_stats["AQI"] / aqi_stats["total_records"] * 100
).round(2)

# Rename columns for clarity
aqi_stats = aqi_stats.rename(columns={"AQI": "missing_aqi_count"})

# Sort by missing percentage in descending order
aqi_stats = aqi_stats.sort_values("missing_percentage", ascending=False)

# Print the results
print("\nMissing AQI values by city in city_day.csv:")
print("==========================================")
for _, row in aqi_stats.iterrows():
    print(f"\nCity: {row['City']}")
    print(f"Total records: {row['total_records']}")
    print(f"Missing AQI values: {row['missing_aqi_count']}")
    print(f"Percentage missing: {row['missing_percentage']}%")

# Save the statistics to a CSV file
output_path = os.path.join(base_dir, "city_day_aqi_missing_stats.csv")
aqi_stats.to_csv(output_path, index=False)
print(f"\nStatistics saved to: {output_path}")

# Print overall statistics
total_missing = aqi_stats["missing_aqi_count"].sum()
total_records = aqi_stats["total_records"].sum()
overall_percentage = (total_missing / total_records * 100).round(2)

print(f"Overall Statistics for city_day.csv:")
print(f"Total records across all cities: {total_records}")
print(f"Total missing AQI values: {total_missing}")
print(f"Overall percentage missing: {overall_percentage}%")

# Additional analysis: Time range for each city
print("\nTime range of data for each city:")
print("================================")
time_range = city_day_df.groupby("City").agg({"Date": ["min", "max"]}).reset_index()

for _, row in time_range.iterrows():
    print(f"\nCity: {row['City']}")
    print(f"First record: {row[('Date', 'min')]}")
    print(f"Last record: {row[('Date', 'max')]}")
