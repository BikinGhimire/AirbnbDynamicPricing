import pandas as pd
from sqlalchemy import create_engine, text

mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ganeshp9',
    'database': 'airbnb'
}

engine = create_engine(f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}/{mysql_config['database']}")

sql_query = """ select * from airbnb.listings_loc;"""
df = pd.read_sql(sql_query,engine)

print(df)

lat =   df.iloc[0]['latitude']
lng = df.iloc[0]['longitude']
print(lat,lng)
import requests

response = requests.get(
  "https://api.locallogic.co/v3/scores",
  headers={
    "Accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJ..."},
  params={
    "lat": lat,
    "lng": lng,
    # "geography_ids": "g30_f25dv0me",
    "geography_levels": "10,20,30",
    "include": "transit_friendly,quiet,cycling_friendly,car_friendly,groceries,nightlife",
    "language": "en",
  }
)

print(response.json())