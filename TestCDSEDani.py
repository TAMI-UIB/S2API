import requests
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

catalogue_odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"

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

for idx, row in result.head(int(n_prods)+1).iterrows():
    product_identifier = row["Id"]
    product_name = row["Name"]

    print(f"\nProcesando producto: {product_name}")

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

    base_dir = Path.home()/'BandesAPP'/product_name
    base_dir.mkdir(exist_ok=True)


    for band_file in tqdm(band_location):
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

        band_path_str = "/".join(band_file)
        if "/R10m/" in band_path_str:
            res_folder = "R10m"
        elif "/R20m/" in band_path_str:
            res_folder = "R20m"
        elif "/R60m/" in band_path_str:
            res_folder = "R60m"
        else:
            res_folder = "other"

        folder_path = base_dir / res_folder
        folder_path.mkdir(exist_ok=True)

        file_name = band_file[-1]
        outfile = folder_path / file_name
        outfile.write_bytes(file.content)
        #print("Saved:", outfile)