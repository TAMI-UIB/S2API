import requests
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from fontTools.misc.cython import returns
from tqdm import tqdm
import urllib3
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def BandDownloader():
    catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    collection_name = "SENTINEL-2"
    product_type = "S2MSI2A"
    max_cloud_cover = "5.00"

    points = []
    while True:
        coordinates = input("Type the coordinates of the AOI one by one (e.g., '30.56 10.89'). When you are done, type 'Done': ")
        if coordinates.lower() == "done":
            break
        points.append(coordinates)

    if len(points) == 1:
        aoi = f"POINT({points[0]})"
    elif len(points) > 1:
        # join coordinates with commas
        closed_points = points + [points[0]]
        aoi = f"POLYGON(({','.join(closed_points)}))"
        print(aoi)
    else:
        aoi = None
        print("No points found")

    #aoi = "POLYGON((20.888443 52.169721,21.124649 52.169721,21.124649 52.271099,20.888443 52.271099,20.888443 52.169721))"
    print(f'The AOI is {aoi}')

    search_period_start = "2024-06-01T00:00:00.000Z"
    search_period_end = "2024-09-30T00:00:00.000Z"

    search_query = (
        f"{catalogue_odata_url}/Products?"
        f"$filter=Collection/Name eq '{collection_name}' and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lT {max_cloud_cover}) and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and "
        f"ContentDate/Start gt {search_period_start} and ContentDate/Start lt {search_period_end}"
    )


    response = requests.get(search_query).json()
    result = pd.DataFrame.from_dict(response["value"])

    n_prods = input(f'A total of {len(result)} products were found. How many do you want to download? ')
    if int(n_prods) > len(result):
        n_prods = input('Venga màquina, posa un número vàlid: ')
    print(f'Downloading the first {n_prods} products.')

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

    for idx, row in result.head(int(n_prods)).iterrows():
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

        # Define which bands to keep for each resolution
        bands_to_keep = {
            "R10m": ["B02", "B03", "B04", "B08", "TCI"],
            "R20m": ["B05", "B06", "B07", "B8A", "B11", "B12"]
        }

        # Filter and annotate band_location with resolution folder
        filtered_bands = []
        for band_file in band_location:
            band_path_str = "/".join(band_file)

            if "/R10m/" in band_path_str and any(band in band_path_str for band in bands_to_keep["R10m"]):
                res_folder = "R10m"
            elif "/R20m/" in band_path_str and any(band in band_path_str for band in bands_to_keep["R20m"]):
                res_folder = "R20m"
            else:
                continue  # skip bands that don't match criteria

            # append a tuple (band_file, res_folder) so we know the folder later
            filtered_bands.append((band_file, res_folder))

        # Now loop over filtered_bands
        for band_file, res_folder in tqdm(filtered_bands):
            tqdm.write(f" 👉 Now processing: {band_file[4]}, {band_file[5]} from {product_name}")

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

    imgs_folder = Path.home() / 'BandesAPP' / session_id
    return imgs_folder

if __name__ == '__main__':
    BandDownloader()