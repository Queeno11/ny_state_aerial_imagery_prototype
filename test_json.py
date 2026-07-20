import json
import pystac_client
from pathlib import Path
c = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
items = list(c.search(collections=["naip"], bbox=[-74.0, 40.7, -73.9, 40.8], max_items=2).items())
payload = {"searched_bbox": [1, 2, 3, 4], "items": [it.to_dict() for it in items]}
print("dumping...")
j = json.dumps(payload)
print("length:", len(j))
Path("test_dump.json").write_text(j)
print("saved")
