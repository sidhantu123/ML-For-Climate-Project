import pandas as pd
import glob
from datetime import datetime
import os
import argparse
from tqdm import tqdm
import numpy as np


def is_valid_measurement(value, measurement_type):
    """Validate measurement values"""
    if pd.isna(value):
        return False
    if not isinstance(value, (int, float)):
        return False
    # Basic negative check as per requirement
    if value < 0:
        return False
    return True


def process_ozone_file(file_path, cities_df):
    """Process a single ozone file with simple averaging"""
    df = pd.read_csv(file_path)

    # Extract date from filename
    filename = os.path.basename(file_path)
    date_parts = filename.replace(".csv", "").split("_")[1:4]
    date = pd.to_datetime(f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}")

    city_data = {}
    grid_points = set(zip(df["lat"], df["lon"]))

    for city in cities_df.index:
        # Get city coordinates
        top_left = eval(cities_df.loc[city, "top_left"])
        bottom_right = eval(cities_df.loc[city, "bottom_right"])
        city_lat = (float(top_left[0]) + float(bottom_right[0])) / 2
        city_lon = (float(top_left[1]) + float(bottom_right[1])) / 2

        # Find nearby points within 1 degree
        nearby_points = []
        for grid_lat, grid_lon in grid_points:
            lon_diff = min(
                abs(grid_lon - city_lon),
                abs(grid_lon - (city_lon + 360)),
                abs(grid_lon - (city_lon - 360)),
            )
            lat_diff = abs(grid_lat - city_lat)

            if lat_diff <= 1.0 and lon_diff <= 1.0:
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                if distance <= 1.0:
                    nearby_points.append((grid_lat, grid_lon))

        if not nearby_points:
            continue

        # Initialize measurement collectors
        measurements = {"TropoO3": [], "OMIO3": [], "StratoO3": []}

        # Collect valid measurements
        for lat, lon in nearby_points:
            point_data = df[(df["lat"] == lat) & (df["lon"] == lon)].iloc[0]

            for measure_type in measurements:
                value = point_data[measure_type]
                if is_valid_measurement(value, measure_type):
                    measurements[measure_type].append(float(value))

        # Calculate simple means for each measurement type
        means = {}
        for measure_type in measurements:
            if measurements[measure_type]:
                means[measure_type] = np.mean(measurements[measure_type])
            else:
                means[measure_type] = None

        # Store results
        city_data[city] = {
            date.strftime("%Y-%m-%d"): {
                "mean_tropo_o3": means["TropoO3"],
                "mean_omi_o3": means["OMIO3"],
                "mean_strato_o3": means["StratoO3"],
            }
        }

    return city_data


def process_year(year, cities_df):
    """Process all files for a given year"""
    print(f"\nProcessing year {year}")

    # Get all files for the year
    file_pattern = f"data/derived_ozone/csv_data/dto_{year}_*.csv"
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"No files found for year {year}")
        return

    print(f"Found {len(files)} files for year {year}")

    # Process all files for the year
    all_city_data = {}
    for file_path in tqdm(files, desc=f"Processing files for {year}"):
        city_data = process_ozone_file(file_path, cities_df)

        # Merge data into all_city_data
        for city, data in city_data.items():
            if city not in all_city_data:
                all_city_data[city] = {}
            all_city_data[city].update(data)

    return all_city_data


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process ozone data for cities within a date range"
    )
    parser.add_argument(
        "--start_year", type=int, required=True, help="Start year (inclusive)"
    )
    parser.add_argument(
        "--end_year", type=int, required=True, help="End year (inclusive)"
    )
    args = parser.parse_args()

    # Validate years
    current_year = datetime.now().year
    if not (
        1900 <= args.start_year <= current_year
        and 1900 <= args.end_year <= current_year
    ):
        print(f"Years must be between 1900 and {current_year}")
        return
    if args.start_year > args.end_year:
        print("Start year must be less than or equal to end year")
        return

    # Read city data
    cities_df = pd.read_csv("data/custom_city_bounding_boxes.csv", index_col=0)

    # Process each year in the range
    all_city_data = {}
    for year in range(args.start_year, args.end_year + 1):
        year_data = process_year(year, cities_df)
        if year_data:
            # Merge data into all_city_data
            for city, data in year_data.items():
                if city not in all_city_data:
                    all_city_data[city] = {}
                all_city_data[city].update(data)

    # Save results to CSV
    if all_city_data:
        # Convert to DataFrame
        rows = []
        for city, dates in all_city_data.items():
            for date, measurements in dates.items():
                row = {
                    "City": city,
                    "Date": date,
                    "TropoO3": measurements["mean_tropo_o3"],
                    "OMIO3": measurements["mean_omi_o3"],
                    "StratoO3": measurements["mean_strato_o3"],
                }
                rows.append(row)

        new_df = pd.DataFrame(rows)
        new_df = new_df.sort_values(["Date", "City"])

        output_file = "data/city_ozone_data.csv"

        # Check if file exists and read existing data
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            # Combine new and existing data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates based on City and Date
            combined_df = combined_df.drop_duplicates(
                subset=["City", "Date"], keep="last"
            )
            # Sort the combined data
            combined_df = combined_df.sort_values(["Date", "City"])
        else:
            combined_df = new_df

        # Save the combined data
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
        print(f"Processed data from {args.start_year} to {args.end_year}")
        print(f"Total unique records: {len(combined_df)}")


if __name__ == "__main__":
    main()
