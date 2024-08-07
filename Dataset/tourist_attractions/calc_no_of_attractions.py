# Importing necessary libraries
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# from ...variables import root_folder

# root_folder = os.path.dirname(__file__)
# print(root_folder)
# exit()
# path = 

# Define downtown coordinates for each city
downtown_coords = {
    'montreal': (45.5017, -73.5673),
    'new_brunswick': (45.9636, -66.6372),  # Fredericton, the capital city of New Brunswick
    'ottawa': (45.4215, -75.6972),
    'quebec_city': (46.8139, -71.2080),
    'toronto': (43.6532, -79.3832),
    'vancouver': (49.2827, -123.1207),
    'victoria': (48.4284, -123.3656),
    'winnipeg': (49.8951, -97.1384)
}
# def haversine_distance(lat1, lon1, lat2, lon2):
#     R = 6371  # Earth's radius in kilometers

#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1

#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * atan2(sqrt(a), sqrt(1-a))
#     distance = R * c

#     return distance
attraction_df = pd.read_csv('canadian_tourist_attractions.csv')
print(attraction_df['city'].unique())
def distance_to_towncenter(lat1,lon1,city):
    lat2,lon2 = downtown_coords[city.lower().replace(' ','_')]
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def haversine_distance(lat1,lon1,lat2,lon2):
    # lat2,lon2 = downtown_coords[city.lower().replace(' ','_')]
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance
    # return dist
def count_attractions_within_radius(airbnb_lat, airbnb_lon, city, radius_km=25):
    # print(city.lower().replace(' ','_'))
    # print(df)
    # print(attraction_df[attraction_df['city']==city.lower().replace(' ','_')])
    attractions_df = attraction_df[attraction_df['city']==city.lower().replace(' ','_')]
    distances = attractions_df.apply(
        lambda row: haversine_distance(airbnb_lat, airbnb_lon, row['latitude'], row['longitude']),
        axis=1
    )
    return sum(distances <= radius_km)
# dist_to_towncenters(1,1,1,'toronto')
# Loading airbnb data of 8 cities
montreal = pd.read_csv('../airbnb/Montreal.csv')
newbrunswick = pd.read_csv('../airbnb/NewBrunswick.csv')
ottawa = pd.read_csv('../airbnb/Ottawa.csv')
quebeccity = pd.read_csv('../airbnb/QuebecCity.csv')
toronto = pd.read_csv('../airbnb/Toronto.csv')
vancouver = pd.read_csv('../airbnb/Vancouver.csv')
victoria = pd.read_csv('../airbnb/Victoria.csv')
winnipeg = pd.read_csv('../airbnb/Winnipeg.csv')

montreal['City'] = 'Montreal'
newbrunswick['City'] = 'New Brunswick'
ottawa['City'] = 'Ottawa'
quebeccity['City'] = 'Quebec City'
toronto['City'] = 'Toronto'
vancouver['City'] = 'Vancouver'
victoria['City'] = 'Victoria'
winnipeg['City'] = 'Winnipeg'

airbnb_df = pd.concat([montreal, newbrunswick, ottawa, quebeccity, toronto, vancouver, victoria, winnipeg], ignore_index=True)
# print(airbnb_df.columns)
df = airbnb_df[['id','latitude','longitude','City']]
df['distance_to_downtown'] = df.apply(
        lambda row: distance_to_towncenter(row['latitude'], row['longitude'],row['City']),
        axis=1
    )
print(df)
df['attractions_within_25km'] = df.apply(
            lambda row: count_attractions_within_radius(row['latitude'], row['longitude'], row['City']),
            axis=1
        )
print(df)
# print(airbnb_df.duplicated('id').sum(0))