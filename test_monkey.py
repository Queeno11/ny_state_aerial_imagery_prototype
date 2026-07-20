import pystac_client
c = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
orig = c._stac_io.session.request
def tr(*a, **kw):
    kw.setdefault('timeout', 30)
    print("requesting with timeout", kw['timeout'])
    return orig(*a, **kw)
c._stac_io.session.request = tr
list(c.search(collections=['naip'], bbox=[-74.0, 40.7, -73.9, 40.8]).items())
print("success")
