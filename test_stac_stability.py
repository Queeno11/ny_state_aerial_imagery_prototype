import pystac_client
c = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
try:
    items = list(c.search(collections=['naip'], bbox=[-74.0, 40.7, -73.9, 40.8]).items())
    print("Success, found", len(items))
except Exception as e:
    print("Failed!", e)
