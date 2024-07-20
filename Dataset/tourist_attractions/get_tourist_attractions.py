import overpy
import pandas as pd

# Initialize Overpass API
api = overpy.Overpass()

# List of cities and their coordinates (latitude, longitude)
cities = {
    'Toronto': (43.65107, -79.347015),
    'Vancouver': (49.282729, -123.120738),
    'Montreal': (45.501688, -73.567255),
    'Calgary': (51.044733, -114.071883),
    'Ottawa': (45.421530, -75.697193),
    'Edmonton': (53.546125, -113.493823),
    'Quebec City': (46.813878, -71.207981),
    'Winnipeg': (49.895136, -97.138374)
}

# Types to exclude
excluded_types = {'hotel', 'camp_site', 'motel', 'guest_house', 'picnic_site','hostel','apartment','information','yes'}

def get_attractions(city, lat, lon):
    # Define the query to fetch tourist attractions within 25km radius
    query = f"""
    [out:json];
    (
      node["tourism"](around:25000,{lat},{lon});
      way["tourism"](around:25000,{lat},{lon});
      relation["tourism"](around:25000,{lat},{lon});
    );
    out center;
    """
    
    result = api.query(query)
    attractions = []
    
    for element in result.nodes + result.ways + result.relations:
        name = element.tags.get('name', '')
        element_type = element.tags.get('tourism', 'attraction')
        if name and element_type not in excluded_types:  # Only include attractions with a name and not in excluded types
            attractions.append({
                'city': city,
                'name': name,
                'latitude': element.lat if hasattr(element, 'lat') else element.center_lat,
                'longitude': element.lon if hasattr(element, 'lon') else element.center_lon,
                'type': element_type,
                'osm_id': element.id
            })
    
    return attractions

# Collect attractions for each city
all_attractions = []

for city, (lat, lon) in cities.items():
    print(f"Fetching attractions for {city}...")
    attractions = get_attractions(city, lat, lon)
    all_attractions.extend(attractions)
    print(f"Found {len(attractions)} attractions in {city}")

# Convert to DataFrame
attractions_df = pd.DataFrame(all_attractions)

# Remove duplicates based on name and coordinates
attractions_df = attractions_df.drop_duplicates(subset=['name', 'latitude', 'longitude'])

# Group attractions by city and type
city_type_counts = attractions_df.groupby(['city', 'type']).size().unstack(fill_value=0)

# Print summary
print("\nSummary of attractions by city and type:")
print(city_type_counts)

# Save to CSV
attractions_df.to_csv('canadian_tourist_attractions_osm.csv', index=False)
print("\nAttractions data saved to 'canadian_tourist_attractions_osm.csv'")

# Additional analysis
print("\nTop 10 types of attractions:")
print(attractions_df['type'].value_counts().head(10))

top_10 = attractions_df['type'].value_counts().head(10)
print(top_10)

print("\nTotal number of unique attractions:", len(attractions_df))
