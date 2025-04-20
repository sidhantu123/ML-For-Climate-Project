import urllib.request
import urllib.error
import urllib.parse
import sys
import time
import os

# List of all cities
CITIES = [
    "Ahmedabad",
    "Aizawl",
    "Amaravati",
    "Amritsar",
    "Bengaluru",
    "Bhopal",
    "Brajrajnagar",
    "Chandigarh",
    "Chennai",
    "Coimbatore",
    "Delhi",
    "Ernakulam",
    "Gurugram",
    "Guwahati",
    "Hyderabad",
    "Jaipur",
    "Jorapokhar",
    "Kochi",
    "Kolkata",
    "Lucknow",
    "Mumbai",
    "Patna",
    "Shillong",
    "Talcher",
    "Thiruvananthapuram",
    "Visakhapatnam",
]

# Define parameters
start_date = "2015-01-01"
end_date = "2020-07-01"
unit_group = "us"  # Use 'metric' for Celsius
include = "days"
api_key = "PLACED_HOLDER"
content_type = "csv"

# Create output directory if it doesn't exist
output_dir = "weather_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def fetch_weather_data(city):
    """Fetch weather data for a specific city"""
    # Encode the location for URL
    encoded_location = urllib.parse.quote(f"{city}, India")

    # Construct the API URL
    api_url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{encoded_location}/{start_date}/{end_date}"
        f"?unitGroup={unit_group}&include={include}&key={api_key}&contentType={content_type}"
    )

    output_file = os.path.join(output_dir, f"{city.lower()}_weather.csv")

    try:
        print(f"Fetching data for {city}...")
        with urllib.request.urlopen(api_url) as response:
            csv_data = response.read()
            with open(output_file, "wb") as file:
                file.write(csv_data)
            print(f"Data saved to '{output_file}'")
        return True
    except urllib.error.HTTPError as e:
        error_info = e.read().decode()
        print(f"HTTP Error for {city}:", e.code, error_info)
        return False
    except urllib.error.URLError as e:
        print(f"URL Error for {city}:", e.reason)
        return False


def main():
    successful_cities = []
    failed_cities = []

    for city in CITIES:
        if fetch_weather_data(city):
            successful_cities.append(city)
        else:
            failed_cities.append(city)
        # Add delay to avoid hitting API rate limits
        time.sleep(1)

    # Print summary
    print("\nSummary:")
    print(f"Successfully fetched data for {len(successful_cities)} cities:")
    for city in successful_cities:
        print(f"- {city}")

    if failed_cities:
        print(f"\nFailed to fetch data for {len(failed_cities)} cities:")
        for city in failed_cities:
            print(f"- {city}")


if __name__ == "__main__":
    main()
