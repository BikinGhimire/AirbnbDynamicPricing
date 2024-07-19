import requests
import os
import gzip
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine,text
from urllib.parse import urljoin
from datetime import datetime,timedelta
# https://data.insideairbnb.com/canada/mb/winnipeg/2024-06-18/data/listings.csv.gz
def execute_queries(engine,query):
    with engine.connect() as conn:
        result = conn.execute(query)
    return result

def save_to_mysql(file_paths, engine,state,city,date):
    try:
        print("Writing into MYSQL database")
        for file_path in file_paths:
            print(file_path)
            # table_name = os.path.splitext(os.path.basename(file_path))[0].replace('-','_')
            table_name = file_path.split('_')[-1].replace('.csv','')
            print(table_name)
            # query = text(f"select count(*) from {table_name};")
            # data = execute_queries(engine,query).fetchone()[0]
            # print(data)
            # if data > 0:
            #     sql_query = text(f""" delete from {table_name} where state = '{state}' and city = '{city}' and run_date = '{date}'; """)
            #     execute_queries(engine,sql_query)
            if file_path.endswith('.csv'):
                chunksize = 100000
                for df in pd.read_csv(file_path, chunksize=chunksize):
                    # print(df)
                    # break
                # df = pd.read_csv(file_path)
                    df['state']=state
                    df['city'] = city
                    df['run_date']=date
                    
                    print(len(df))
                    df.to_sql(table_name, engine, if_exists='append', index=False)
                print(f"Saved {file_path} to MySQL table: {table_name}")
            elif file_path.endswith('.geojson'):
                # For geojson files, you might need to use a different approach
                print(f"Skipping {file_path} as it's a geojson file")
    except Exception as e:
        exit(e)
        

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        print(f"{local_filename} found")
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_and_extract_gzip(url, local_filename):
    
    gzip_filename = local_filename + '.gz'
    download_file(url, gzip_filename)
    with gzip.open(gzip_filename, 'rb') as f_in:
        with open(local_filename, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(gzip_filename)

def download_airbnb_data(base_url, state,city, date):
    while str(date) >= '2023-01-01':
        print(state,city,date)
        city_url = urljoin(base_url, f"{state}/{city}/{date}/data/")
        files = [
            "listings.csv.gz",
            "calendar.csv.gz",
            "reviews.csv.gz",
            "listings.csv",
            "reviews.csv",
            # "neighbourhoods.csv",
            # "neighbourhoods.geojson"
        ]

        try:
            file_paths=[]
            for file in files:
                file_url = urljoin(city_url, file)
                # print(file_url)
                # break
                local_filename = f"{city}_{file.replace('.gz', '')}"
                # print(f"Downloading {file} for {city} for {date}...")
                
                if file.endswith('.gz'):
                    download_and_extract_gzip(file_url, local_filename)
                    file_paths.append(local_filename)
                else:
                    download_file(file_url, local_filename.replace('.csv','summary.csv'))
                    file_paths.append( local_filename.replace('.csv','summary.csv'))

            print(f"Finished downloading data for {city}")
            # file_paths = [f"{city}_{file.replace('.gz', '')}" for file in files]
            # print(file_paths)
            save_to_mysql(file_paths, engine, state,city,date)
            # return [f"{city}_{file.replace('.gz', '')}" for file in files]
        except Exception as e:
            error = str(e)
            if 'Forbidden for url' in error:
                date = date- timedelta(days=1)
                if str(date)>='2023-06-01':
                    download_airbnb_data(base_url,state, city, date)
                else:
                    exit
            else: 
                raise e
        date = date- timedelta(days=1)



# MySQL connection settings
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ganeshp9',
    'database': 'airbnb'
}

# Create MySQL engine
engine = create_engine(f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}/{mysql_config['database']}")
# https://data.insideairbnb.com/canada/bc/vancouver/2024-06-15/data/listings.csv.gz
# Main execution
base_url = "http://data.insideairbnb.com/canada/"
canadian_cities = {"vancouver", "toronto", "montreal", "quebec-city", "ottawa","new-brunswick","victoria","winnipeg"}
# canadian_cities= {"ottawa"}
# date = "2024-06-18"  # Update this to the most recent date available
date = datetime.today().date()
city_details = {
"vancouver":
    {"city":"vancouver",
    "state":"bc"
        },
"toronto":{
    "city":"toronto",
    "state":"on"
        },
"montreal":{
        "city":"montreal",
        "state":"qc"
    },
"quebec-city":{"city":"quebec-city",
    "state":"qc"
        },
"ottawa":{"city":"ottawa",
    "state":"on"
        },
"new-brunswick":{"city":"new-brunswick",
    "state":"nb"
        },
"victoria":{
        "city":"victoria",
        "state":"bc"
    }   ,
"winnipeg":{
        "city":"winnipeg",
        "state":"mb"
    } 
}
for city1 in canadian_cities:
    state = city_details[city1].get('state')
    city = city_details[city1].get('city')
    file_paths = download_airbnb_data(base_url,state,city, date)
    # save_to_mysql(file_paths, engine)

print("All data downloaded and saved to MySQL!")
