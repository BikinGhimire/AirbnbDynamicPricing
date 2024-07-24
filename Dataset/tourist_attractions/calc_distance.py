import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import warnings

warnings.filterwarnings('ignore')
import time

# Start timing the entire program
start_time = time.time()

# Define downtown coordinates for each city
downtown_coords = {
    'montreal': (45.5017, -73.5673),
    'new_brunswick': (45.9636, -66.6372),  # Fredericton
    'ottawa': (45.4215, -75.6972),
    'quebec_city': (46.8139, -71.2080),
    'toronto': (43.6532, -79.3832),
    'vancouver': (49.2827, -123.1207),
    'victoria': (48.4284, -123.3656),
    'winnipeg': (49.8951, -97.1384)
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Load attraction data
attraction_df = pd.read_csv('canadian_tourist_attractions.csv')

# Load Airbnb data
cities = ['Montreal', 'NewBrunswick', 'Ottawa', 'QuebecCity', 'Toronto', 'Vancouver', 'Victoria', 'Winnipeg']
airbnb_df = pd.concat([pd.read_csv(f'../airbnb/{city}.csv').assign(City=city) for city in cities], ignore_index=True)

# Prepare the data
df = airbnb_df[['id', 'latitude', 'longitude', 'City']].copy()
df['City'] = df['City'].str.lower().str.replace(' ', '_')

# Calculate distance to downtown using broadcasting
for city, (lat, lon) in downtown_coords.items():
    mask = df['City'] == city
    df.loc[mask, 'distance_to_downtown'] = haversine_distance(
        df.loc[mask, 'latitude'].values, 
        df.loc[mask, 'longitude'].values, 
        lat, 
        lon
    )

# Count attractions within 25km using broadcasting
def count_attractions_within_radius(airbnb_lat, airbnb_lon, attractions, radius_km=25):
    distances = haversine_distance(airbnb_lat, airbnb_lon, attractions['latitude'].values, attractions['longitude'].values)
    return np.sum(distances <= radius_km)

# Calculate attractions within 25km for each Airbnb listing
attraction_counts = []
for city in df['City'].unique():
    attractions_city = attraction_df[attraction_df['city'] == city]
    if not attractions_city.empty:
        attraction_counts.extend(df[df['City'] == city].apply(
            lambda row: count_attractions_within_radius(row['latitude'], row['longitude'], attractions_city),
            axis=1
        ))
    else:
        attraction_counts.extend([0] * len(df[df['City'] == city]))

df['attractions_within_25km'] = attraction_counts

print(df.head())
print(len(df))
final_df = airbnb_df.merge(df,'inner',on=['id', 'latitude', 'longitude'])
final_df.drop('City_y',axis=1,inplace=True)
final_df.rename({'City_x':'City'},axis=1,inplace=True)
final_df['distance_to_downtown']=round(final_df['distance_to_downtown'],2)
print(final_df.head())
print(len(final_df))
cities = ['Montreal', 'NewBrunswick', 'Ottawa', 'QuebecCity', 'Toronto', 'Vancouver', 'Victoria', 'Winnipeg']
for city in cities:
    final_df[final_df['City']==city].to_csv(f'../airbnb/{city}.csv',header=True,index=False)
# Calculate and print total runtime
end_time = time.time()
total_runtime = end_time - start_time
print(f"\nTotal runtime: {total_runtime:.2f} seconds")

print(df[df['City']=='montreal'].sort_values('attractions_within_25km'))