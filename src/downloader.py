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
import sys
from tkinter import simpledialog
from tkinter import filedialog

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def BandDownloader(aoi_points, search_period_start, search_period_end,
                   max_cloud_cover, progress_var, progress_bar):
    global search_results, catalogue_odata_url, session_id
    catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build AOI
    if len(aoi_points) == 1:
        if aoi_points[0] == '':
            messagebox.showerror("Error", "No coordinates provided. Please enter at least one point.")
            return
        else:
            aoi = f"POINT({aoi_points[0]})"
    elif len(aoi_points) > 1:
        closed_points = aoi_points + [aoi_points[0]]
        aoi = f"POLYGON(({','.join(closed_points)}))"
    else:
        messagebox.showerror("Error", "Invalid coordinates format.")
        return

    # Search query
    collection_name = "SENTINEL-2"
    product_type = "S2MSI2A"

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

    if result.shape[0] == 0:
        messagebox.showinfo("No products", "No products matching criteria found.")
        return

    # Ask user how many to download
    total_found = result.shape[0]

    show_confirm_frame(total_found)
    # question_label.config(text=f"Found {total_found} product(s). How many do you want (≤ {total_found})?")
    #
    # nprods_var.set(total_found)
    #
    # # Enable the download button
    # confirm_button.config(state="normal")
    #
    # # Save results globally so confirm_download can access them
    global search_results
    search_results = result

def confirm_download():
    try:
        n_prods = int(nprods_var.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number.")
        return

    total_found = search_results.shape[0]
    if n_prods < 1 or n_prods > total_found:
        messagebox.showerror("Error", f"Enter a number between 1 and {total_found}.")
        return

    question_label.config(text=f"Downloading {n_prods} of {total_found} products...")
    confirm_frame.grid_forget()

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

    total_bands = 13

    progress_bar['maximum'] = total_bands
    #downloaded = 0

    # Start downloading bands
    for idx, row in search_results.head(int(n_prods)).iterrows():
        print("Downloading product " + str(idx+1) + "...")
        downloaded = 0
        progress_var.set(downloaded)
        progress_bar.update()

        product_identifier = row["Id"]
        product_name = row["Name"]

        # XML metadata
        url = f"{catalogue_odata_url}/Products({product_identifier})/Nodes({product_name})/Nodes(MTD_MSIL2A.xml)/$value"
        r = session.get(url, allow_redirects=False)
        while r.status_code in (301, 302, 303, 307):
            url = r.headers["Location"]
            r = session.get(url, allow_redirects=False)
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

        base_path = Path(save_dir_var.get().strip()) if save_dir_var.get().strip() else Path.home() / "BandesAPP"
        base_dir = base_path / str(session_id) / product_name
        base_dir.mkdir(parents=True, exist_ok=True)

        print(f"Created {base_dir} directory")

        bands_to_keep = {
            "R10m": ["B02", "B03", "B04", "B08", "TCI"],
            "R20m": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "R60m": ["B01", "B09"]
        }

        filtered_bands = []
        for band_file in band_location:
            band_path_str = "/".join(band_file)
            if "/R10m/" in band_path_str and any(b in band_path_str for b in bands_to_keep["R10m"]):
                res_folder = "R10m"
            elif "/R20m/" in band_path_str and any(b in band_path_str for b in bands_to_keep["R20m"]):
                res_folder = "R20m"
            elif "/R60m/" in band_path_str and any(b in band_path_str for b in bands_to_keep["R60m"]):
                res_folder = "R60m"
            else:
                continue
            filtered_bands.append((band_file, res_folder))

        for band_file, res_folder in filtered_bands:
            url = f"{catalogue_odata_url}/Products({product_identifier})/Nodes({product_name})"
            for node in band_file[1:]:
                url += f"/Nodes({node})"
            url += "/$value"

            r = session.get(url, allow_redirects=False)
            while r.status_code in (301, 302, 303, 307):
                url = r.headers["Location"]
                r = session.get(url, allow_redirects=False)
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
    return session_id


# ---------------- GUI ----------------
def run_downloader():
    coords = coords_entry.get("1.0", tk.END).strip().split("\n")
    start_date = start_entry.get().strip() + "T00:00:00.000Z"
    end_date = end_entry.get().strip() + "T23:59:59.000Z"
    max_cloud = cloud_entry.get()

    # Run in a thread so GUI stays responsive
    threading.Thread(
        target=BandDownloader,
        args=(coords, start_date, end_date, max_cloud, progress_var, progress_bar),
        daemon=True
    ).start()


# GUI setup
root = tk.Tk()
root.title("Sentinel-2 Band Downloader")
save_dir_var = tk.StringVar(value=str(Path.home() / "BandesAPP"))

def choose_directory():
    folder = filedialog.askdirectory()
    if folder:
        save_dir_var.set(folder)

# Coordinates input
ttk.Label(root, text="Coordinates (one per line)").grid(row=0, column=0, sticky="w")
coords_entry = tk.Text(root, width=40, height=5)
coords_entry.grid(row=1, column=0, padx=5, pady=5)

# Save directory
ttk.Label(root, text="Save directory").grid(row=2, column=0, sticky="w")
save_frame = ttk.Frame(root)
save_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2)

save_entry = ttk.Entry(save_frame, textvariable=save_dir_var, width=29)
save_entry.pack(side="left", fill="x", expand=True)

browse_button = ttk.Button(save_frame, text="Browse", command=choose_directory)
browse_button.pack(side="left", padx=(5, 0))

# Dates section
ttk.Label(root, text="Which dates do you want to consider?").grid(row=4, column=0, sticky="w", padx=5, pady=(10,2))

dates_frame = ttk.Frame(root)
dates_frame.grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)

ttk.Label(dates_frame, text="Start:").pack(side="left")
start_entry = ttk.Entry(dates_frame, width=12)
start_entry.insert(0, "2024-06-01")
start_entry.pack(side="left", padx=(2, 10))

ttk.Label(dates_frame, text="End:").pack(side="left")
end_entry = ttk.Entry(dates_frame, width=12)
end_entry.insert(0, "2024-09-30")
end_entry.pack(side="left", padx=2)


# Cloud cover
cloud_entry = tk.DoubleVar(value=5.00)

ttk.Label(root, text="Max cloud cover (%)").grid(row=8, column=0, pady=10, sticky="w")

frame = ttk.Frame(root)
frame.grid(row=9, column=0, columnspan=2, sticky="w", padx=5)

cloud_slider = tk.Scale(
    frame,
    from_=0, to=100,
    orient="horizontal",
    length=200,
    resolution=0.01,
    variable=cloud_entry,
    showvalue=False,
)
cloud_slider.pack(side="left")

cloud_aux = ttk.Entry(frame, textvariable=cloud_entry, width=8)
cloud_aux.pack(side="left", padx=(5, 0))

# Number of products
# ttk.Label(root, text="Number of products to download").grid(row=8, column=0, sticky="w")
# nprods_entry = ttk.Entry(root, width=30)
# nprods_entry.insert(0, "1")
# nprods_entry.grid(row=9, column=0, padx=5, pady=2)

question_label = ttk.Label(root, text='Click "Run downloader" to look for products')
question_label.grid(row=11, column=0, sticky="w", pady=5)

# Frame to hold number entry + confirm button
confirm_frame = ttk.Frame(root)
# confirm_frame.grid(row=12, column=0, sticky="w", padx=5, pady=5)

nprods_var = tk.StringVar()
nprods_entry = ttk.Entry(confirm_frame, textvariable=nprods_var, width=10)
nprods_entry.pack(side="left")

confirm_button = ttk.Button(confirm_frame, text="Set number", command=confirm_download, state="disabled")
confirm_button.pack(side="left", padx=(5,0))

def show_confirm_frame(total_found):
    # Update label / info if needed
    question_label.config(text=f"Found {total_found} product(s). Enter how many to download:")
    # Pre-fill entry with total number
    nprods_var.set(total_found)
    # Enable the button
    confirm_button.config(state="normal")
    # Show the frame
    confirm_frame.grid(row=12, column=0, sticky="w", padx=5, pady=5)

# Frame to hold Run button and progress bar
action_frame = ttk.Frame(root)
action_frame.grid(row=10, column=0, columnspan=2, pady=12, sticky="we", padx=5)

# Run button on the left
run_button = ttk.Button(action_frame, text="Run Downloader", command=run_downloader)
run_button.pack(side="left")

# Progress bar on the right
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(action_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var)
progress_bar.pack(side="right")

root.mainloop()