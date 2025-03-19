import requests

# Define the place and your username
place = "Ahmedabad"
username = "HetviPatel"  # Replace with your GeoNames username

# API URL
url = f"http://api.geonames.org/searchJSON?q={place}&maxRows=1&username={username}"

# Send request
response = requests.get(url)
data = response.json()

# Extract latitude and longitude
if "geonames" in data and len(data["geonames"]) > 0:
    lat = data["geonames"][0]["lat"]
    lon = data["geonames"][0]["lng"]
    print(f"Coordinates of {place}: Latitude = {lat}, Longitude = {lon}")
else:
    print("Place not found")
