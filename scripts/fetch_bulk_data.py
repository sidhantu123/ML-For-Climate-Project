import requests
from bs4 import BeautifulSoup
from io import BytesIO
import zipfile
import os
import xarray as xr

COOKIE = ""


# Function to convert .nc file to CSV
def convert_nc_to_csv(nc_file_path, output_dir, year, month, day):
    # # Read the .nc file
    # dataset = netCDF4.Dataset(nc_file_path, 'r')

    # # Extract data (you may need to change variable names based on the .nc file's structure)
    # variables = list(dataset.variables)

    # # Assuming the dataset has a variable 'data' (modify as needed)
    # data = dataset.variables[variables[0]][:]

    # # Convert to pandas DataFrame
    # df = pd.DataFrame(data)

    ds = xr.open_dataset(nc_file_path)

    # Convert to DataFrame (flatten all dimensions)
    df = ds.to_dataframe().reset_index()

    # Generate CSV file name
    csv_file_path = os.path.join(output_dir, f"dto_{year}_{month}_{day}.csv")

    # Save DataFrame as CSV
    df.to_csv(csv_file_path, index=False)
    # print(f"CSV saved to {csv_file_path}")


# Function to fetch and process data for a given year, month, and day
def fetch_and_process_data(year, month, day):
    # Step 1: Construct the URL and headers
    initial_url = f"https://bhuvan-app3.nrsc.gov.in/data/download/tools/download1/downloadlink.php?id=nices_dto_{year}{month}{day}&se=NICES&sf=dto"

    headers = {
        "accept": "text/html, */*; q=0.01",
        "accept-language": "en-US,en;q=0.9",
        "priority": "u=1, i",
        "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Linux"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "cookie": COOKIE,
        "Referer": "https://bhuvan-app3.nrsc.gov.in/data/download/index.php?c=p&s=NICES&p=cm4&g=AS",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }

    # Step 2: Send GET request to get HTML content
    print(f"Sending request to {initial_url}...")
    response = requests.get(initial_url, headers=headers)
    response.raise_for_status()

    # Step 3: Parse HTML to get the download link
    soup = BeautifulSoup(response.text, "html.parser")
    iframe = soup.find("iframe")

    if not iframe or "src" not in iframe.attrs:
        print("Download URL not found in iframe.")
        return

    zip_download_url = iframe["src"]
    # print(f"ZIP download URL found: {zip_download_url}")

    # Step 4: Download the ZIP file
    # print("Downloading ZIP file...")
    zip_response = requests.get(zip_download_url, headers=headers)
    zip_response.raise_for_status()

    # Step 5: Extract the ZIP file
    extract_to = f"data/derived_ozone/extracted_files_{year}_{month}_{day}"
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(BytesIO(zip_response.content)) as zip_file:
        zip_file.extractall(extract_to)
        # print(f"ZIP file extracted to: {extract_to}")

    # Step 6: Find the .nc file in the extracted folder
    nc_files = [f for f in os.listdir(extract_to) if f.endswith(".nc")]
    csv_data = "data/derived_ozone/csv_data"
    if nc_files:
        nc_file_path = os.path.join(extract_to, nc_files[0])
        # Step 7: Convert the .nc file to CSV
        convert_nc_to_csv(nc_file_path, csv_data, year, month, day)


# Loop through 2015-2020 for all months
for year in range(2019, 2021):
    for month in range(1, 13):
        for day in range(1, 32):  # Handling all possible days
            # Format month and day to two digits
            month_str = f"{month:02d}"
            day_str = f"{day:02d}"

            try:
                fetch_and_process_data(str(year), month_str, day_str)
            except Exception as e:
                print(f"Failed to fetch data for {year}-{month_str}-{day_str}: {e}")
