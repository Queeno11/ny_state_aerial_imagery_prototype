import pystac_client
print(pystac_client.__version__)
c = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
print(hasattr(c, '_stac_io'))
if hasattr(c, '_stac_io'):
    print(type(c._stac_io))
    print(hasattr(c._stac_io, 'session'))
