import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
import threading
import requests
import json
import xml.etree.ElementTree as ET
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def BandDownloader(aoi_points, max_cloud_cover, n_prods, progress_var, progress_bar):
    catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build AOI
    if len(aoi_points) == 1:
        aoi = f"POINT({aoi_points[0]})"
    elif len(aoi_points) > 1:
        closed_points = aoi_points + [aoi_points[0]]
        aoi = f"POLYGON(({','.join(closed_points)}))"
    else:
        raise ValueError("No points provided")

    # Search query
    collection_name = "SENTINEL-2"
    product_type = "S2MSI2A"
    search_period_start = "2024-06-01T00:00:00.000Z"
    search_period_end = "2024-09-30T00:00:00.000Z"

    search_query = (
        f"{catalogue_odata_url}/Products?"
        f"$filter=Collection/Name eq '{collection_name}' and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover}) and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and "
        f"ContentDate/Start gt {search_period_start} and ContentDate/Start lt {search_period_end}"
    )

    response = requests.get(search_query).json()
    result = pd.DataFrame.from_dict(response["value"])

    # Authentication
    username = "francesc.alcover@uib.cat"
    password = "Dani2000pato!"
    auth_server_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    response = requests.post(auth_server_url, data=data, verify=True, allow_redirects=False)
    access_token = json.loads(response.text)["access_token"]
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {access_token}"

    total_bands = 0
    # First, count how many bands will be downloaded
    for idx, row in result.head(int(n_prods) + 1).iterrows():
        product_name = row["Name"]
        url = f"{catalogue_odata_url}/Products({row['Id']})/Nodes({product_name})/Nodes(MTD_MSIL2A.xml)/$value"
        response = session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers["Location"]
            response = session.get(url, allow_redirects=False)
        file = session.get(url, verify=False, allow_redirects=True)
        outfile = Path.home() / f"{product_name}_MTD_MSIL2A.xml"
        outfile.write_bytes(file.content)
        tree = ET.parse(str(outfile))
        root = tree.getroot()
        for granule in root.findall(".//Granule"):
            for img_file in granule.findall(".//IMAGE_FILE"):
                band_path = f"{product_name}/{img_file.text}.jp2".split("/")
                # Only keep desired bands
                if any(b in band_path[-1] for b in
                       ["B02", "B03", "B04", "B08", "TCI", "B05", "B06", "B07", "B8A", "B11", "B12"]):
                    total_bands += 1

    progress_bar['maximum'] = total_bands
    downloaded = 0

    # Start downloading bands
    for idx, row in result.head(int(n_prods) + 1).iterrows():
        product_identifier = row["Id"]
        product_name = row["Name"]
        url = f"{catalogue_odata_url}/Products({product_identifier})/Nodes({product_name})/Nodes(MTD_MSIL2A.xml)/$value"

        response = session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers["Location"]
            response = session.get(url, allow_redirects=False)
        file = session.get(url, verify=False, allow_redirects=True)
        outfile = Path.home() / f"{product_name}_MTD_MSIL2A.xml"
        outfile.write_bytes(file.content)
        tree = ET.parse(str(outfile))
        root = tree.getroot()
        band_location = []
        for granule in root.findall(".//Granule"):
            for img_file in granule.findall(".//IMAGE_FILE"):
                band_path = f"{product_name}/{img_file.text}.jp2".split("/")
                band_location.append(band_path)

        base_dir = Path.home() / 'BandesAPP' / str(session_id) / product_name
        base_dir.mkdir(parents=True, exist_ok=True)

        bands_to_keep = {
            "R10m": ["B02", "B03", "B04", "B08", "TCI"],
            "R20m": ["B05", "B06", "B07", "B8A", "B11", "B12"]
        }

        filtered_bands = []
        for band_file in band_location:
            band_path_str = "/".join(band_file)
            if "/R10m/" in band_path_str and any(b in band_path_str for b in bands_to_keep["R10m"]):
                res_folder = "R10m"
            elif "/R20m/" in band_path_str and any(b in band_path_str for b in bands_to_keep["R20m"]):
                res_folder = "R20m"
            else:
                continue
            filtered_bands.append((band_file, res_folder))

        for band_file, res_folder in filtered_bands:
            url = f"{catalogue_odata_url}/Products({product_identifier})/Nodes({product_name})"
            for node in band_file[1:]:
                url += f"/Nodes({node})"
            url += "/$value"

            response = session.get(url, allow_redirects=False)
            while response.status_code in (301, 302, 303, 307):
                url = response.headers["Location"]
                response = session.get(url, allow_redirects=False)
            file = session.get(url, verify=False, allow_redirects=True)

            folder_path = base_dir / res_folder
            folder_path.mkdir(exist_ok=True)
            file_name = band_file[-1]
            outfile = folder_path / file_name
            outfile.write_bytes(file.content)

            downloaded += 1
            progress_var.set(downloaded)
            progress_bar.update()

    messagebox.showinfo("Done", f"Download complete! Files saved in {Path.home() / 'BandesAPP' / session_id}")


# ---------------- GUI ----------------
import tkinter as tk
from tkinter import ttk
from datetime import datetime

def run_downloader():
    coords = coords_entry.get("1.0", tk.END).strip().split("\n")
    start_date = start_entry.get().strip()
    end_date = end_entry.get().strip()
    max_cloud = cloud_entry.get().strip()
    n_prods = nprods_entry.get().strip()

    print("Coordinates:", coords)
    print("Search start:", start_date)
    print("Search end:", end_date)
    print("Max cloud cover:", max_cloud)
    print("Number of products to download:", n_prods)

    # ⚡️ Here you call your BandDownloader, passing start_date, end_date, n_prods, etc.
    # BandDownloader(coords, start_date, end_date, max_cloud, n_prods)

# GUI setup
root = tk.Tk()
root.title("Sentinel-2 Band Downloader")

# Coordinates input
ttk.Label(root, text="Coordinates (one per line)").grid(row=0, column=0, sticky="w")
coords_entry = tk.Text(root, width=40, height=5)
coords_entry.grid(row=1, column=0, padx=5, pady=5)

# Start date
ttk.Label(root, text="Search start date (YYYY-MM-DD)").grid(row=2, column=0, sticky="w")
start_entry = ttk.Entry(root, width=30)
start_entry.insert(0, "2024-06-01")  # default
start_entry.grid(row=3, column=0, padx=5, pady=2)

# End date
ttk.Label(root, text="Search end date (YYYY-MM-DD)").grid(row=4, column=0, sticky="w")
end_entry = ttk.Entry(root, width=30)
end_entry.insert(0, "2024-09-30")  # default
end_entry.grid(row=5, column=0, padx=5, pady=2)

# Cloud cover
ttk.Label(root, text="Max cloud cover (%)").grid(row=6, column=0, sticky="w")
cloud_entry = ttk.Entry(root, width=30)
cloud_entry.insert(0, "5.00")
cloud_entry.grid(row=7, column=0, padx=5, pady=2)

# Number of products
ttk.Label(root, text="Number of products to download").grid(row=8, column=0, sticky="w")
nprods_entry = ttk.Entry(root, width=30)
nprods_entry.insert(0, "1")  # default = 1
nprods_entry.grid(row=9, column=0, padx=5, pady=2)

# Run button
run_button = ttk.Button(root, text="Run Downloader", command=run_downloader)
run_button.grid(row=10, column=0, pady=10)

root.mainloop()