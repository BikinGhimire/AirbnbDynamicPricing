# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split, RepeatedKFold
from scipy import stats
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from numpy import logspace
import lightgbm as lgb
import xgboost as xgb
from sklearn.cluster import KMeans
from geopy.distance import great_circle
import shap
import warnings
warnings.filterwarnings('ignore')


# Loading airbnb data of 8 cities and merging them into a single dataframe
montreal = pd.read_csv('Dataset/Airbnb/Montreal.csv')
newbrunswick = pd.read_csv('Dataset/Airbnb/NewBrunswick.csv')
ottawa = pd.read_csv('Dataset/Airbnb/Ottawa.csv')
quebeccity = pd.read_csv('Dataset/Airbnb/QuebecCity.csv')
toronto = pd.read_csv('Dataset/Airbnb/Toronto.csv')
vancouver = pd.read_csv('Dataset/Airbnb/Vancouver.csv')
victoria = pd.read_csv('Dataset/Airbnb/Victoria.csv')
winnipeg = pd.read_csv('Dataset/Airbnb/Winnipeg.csv')

montreal['city'] = 'Montreal'
newbrunswick['city'] = 'New Brunswick'
ottawa['city'] = 'Ottawa'
quebeccity['city'] = 'Quebec City'
toronto['city'] = 'Toronto'
vancouver['city'] = 'Vancouver'
victoria['city'] = 'Victoria'
winnipeg['city'] = 'Winnipeg'

airbnb_df = pd.concat([montreal, newbrunswick, ottawa, quebeccity, toronto, vancouver, victoria, winnipeg],
                      ignore_index=True)

# Loading the review sentiment scores and merging them into airbnb dataframe
reviews_df = pd.read_csv('Dataset/Sentiment/listing_sentiment_scores.csv')
airbnb_df = pd.merge(airbnb_df, reviews_df, left_on='id', right_on='listing_id', how='left')

# Eliminating initial features
feature_elimination_list = [
    'listing_url', 'scrape_id', 'last_scraped', 'source', 'host_url', 'host_thumbnail_url', 'calendar_updated',
    'calendar_last_scraped', 'host_name', 'listing_id', 'host_id', 'neighbourhood_group_cleansed',
    'host_neighbourhood', 'host_location', 'neighbourhood', 'host_listings_count', 'host_total_listings_count',
    'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms', 'reviews_per_month', 'review_scores_accuracy', 'bathrooms_text',
    'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
    'host_about', 'neighborhood_overview', 'host_has_profile_pic', 'picture_url', 'host_picture_url',
    'calculated_host_listings_count', 'number_of_reviews'
]
airbnb_df = airbnb_df.drop(feature_elimination_list, axis=1)

# Loading the tourist attractions data
attractions_df = pd.read_csv('Dataset/Locations/canadian_tourist_attractions.csv')
attractions_df['city'] = attractions_df['city'].str.title().str.replace('_', ' ')
attractions_df.head()

# Mapping new column names
column_mappings = {
    'name': 'title',
    'first_review': 'first_review_date',
    'last_review': 'last_review_date',
    'review_scores_value': 'review_scores_value_for_money',
    'neighbourhood_cleansed': 'neighbourhood',
    'sentiment_score': 'review_sentiment_score'
}
airbnb_df.rename(columns=column_mappings, inplace=True)

# Removing "$" from price and converting to float
airbnb_df['price'] = airbnb_df['price'].str.replace('[$,]', '', regex=True).astype(float)


################ DATA PREPROCESSING ################

# Splitting the data into training and testing sets. High testing size to reduce overfitting.
data_train, data_test = train_test_split(airbnb_df, test_size=0.01, random_state=42)

# Converting values in price column to its natural logarithm
data_train['price'] = np.log(data_train['price'])

# Filling null value with unlicensed
data_train['license'].fillna('Unlicensed', inplace=True)

# Drop rows with empty price
data_train = data_train.dropna(subset=['price'])

# Dropping records with null reviews so that we have more accurate prediction
data_train = data_train.dropna(subset=['review_sentiment_score'])
data_train = data_train.dropna(subset=['review_scores_rating'])

# Filling null values in host_is_superhost and 'availability' with "f" since the missing values corresponds to none.
data_train['host_is_superhost'] = data_train['host_is_superhost'].fillna('f')
data_train['has_availability'] = data_train['has_availability'].fillna('f')

# Filling null values in host_response_time to mode of the feature
data_train['host_response_time'] = data_train['host_response_time'].fillna(data_train['host_response_time'].mode()[0])

# Remove % sign and convert to numeric for the following columns
data_train['host_response_rate'] = pd.to_numeric(data_train['host_response_rate'].str.replace('%', ''))
data_train['host_acceptance_rate'] = pd.to_numeric(data_train['host_acceptance_rate'].str.replace('%', ''))

# Filling null values in remaining columns with median
numeric_columns = [
    'host_response_rate',
    'host_acceptance_rate',
    'bedrooms', 'beds',
    'review_scores_value_for_money',
    'review_scores_location',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_cleanliness',
    'bathrooms'
]
for column in numeric_columns:
    median_value = data_train[column].median()
    data_train[column].fillna(median_value, inplace=True)


################ FEATURE ENGINEERING ################

# Adding days since columns using the date columns
current_date = datetime.now()
data_train['host_since'] = pd.to_datetime(data_train['host_since'])
data_train['host_since_days'] = (current_date - data_train['host_since']).dt.days
data_train.drop(columns=['host_since', 'first_review_date', 'last_review_date'], inplace=True)

# Define downtown coordinates for each city
downtown_coords = {
    'Montreal': (45.5017, -73.5673),
    'New Brunswick': (45.9636, -66.6372),
    'Ottawa': (45.4215, -75.6972),
    'Quebec City': (46.8139, -71.2080),
    'Toronto': (43.6532, -79.3832),
    'Vancouver': (49.2827, -123.1207),
    'Victoria': (48.4284, -123.3656),
    'Winnipeg': (49.8951, -97.1384)
}

# Function to calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate distance to downtown using broadcasting
for city, (lat, lon) in downtown_coords.items():
    mask = data_train['city'] == city
    data_train.loc[mask, 'distance_to_downtown'] = haversine_distance(
        data_train.loc[mask, 'latitude'].values,
        data_train.loc[mask, 'longitude'].values,
        lat,
        lon
    )

# Function to count attractions within 10km using broadcasting
def count_attractions_within_radius(airbnb_lat, airbnb_lon, attractions, radius_km=10):
    distances = haversine_distance(airbnb_lat, airbnb_lon, attractions['latitude'].values, attractions['longitude'].values)
    return np.sum(distances <= radius_km)

# Calculate attractions within 10km for each Airbnb listing
attraction_counts = []
for city in data_train['city'].unique():
    attractions_city = attractions_df[attractions_df['city'] == city]
    if not attractions_city.empty:
        attraction_counts.extend(data_train[data_train['city'] == city].apply(
            lambda row: count_attractions_within_radius(row['latitude'], row['longitude'], attractions_city),
            axis=1
        ))
    else:
        attraction_counts.extend([0] * len(data_train[data_train['city'] == city]))
data_train['nearby_attractions'] = attraction_counts

# Boolean encoding specific columns
for boolean_column in ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
    data_train[boolean_column] = data_train[boolean_column].map(lambda s: False if s == "f" else True)

# Changing data in license column to licensed and unlicensed and boolean encoding the data
data_train['license'] = data_train['license'].map(lambda s: False if s == "Unlicensed" else True)

# Ordinal Encoding host_response_time since there is a clear order
categories = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']
ordinal_encoder = OrdinalEncoder(categories=[categories])
data_train['host_response_time_encoded'] = ordinal_encoder.fit_transform(data_train[['host_response_time']])
data_train.drop(columns=['host_response_time'], inplace=True)

# One hot encoding property_type, city, and room_type, but keeping the original categorical columns to use
# later on our final selected model
dummy_vars = pd.get_dummies(data_train['property_type'], prefix='property')
data_train = pd.concat([data_train, dummy_vars], axis=1)
dummy_vars = pd.get_dummies(data_train['city'], prefix='city')
data_train = pd.concat([data_train, dummy_vars], axis=1)
dummy_vars = pd.get_dummies(data_train['room_type'], prefix='room_type')
data_train = pd.concat([data_train, dummy_vars], axis=1)

# One hot encoding host_verifications by extracting data stored in a list in the column first
# Function to safely evaluate strings
def safe_eval(x):
    if isinstance(x, str):
        return eval(x)
    return x
# Extract unique items from the list in the column
unique_items = set(item for sublist in data_train['host_verifications'].apply(safe_eval) for item in sublist)
# Apply one hot encoding to the unique_items and create a new column for each item
for item in unique_items:
    column_name = item + "_verification"
    data_train[column_name] = data_train['host_verifications'].apply(lambda x: True if item in x else False)
# Drop the original amenities column
data_train.drop('host_verifications', axis=1, inplace=True)

# Extracting amenities, changing data using regex and onehot encoding the data
unique_items = set(item for sublist in data_train['amenities'].apply(safe_eval) for item in sublist if item)
#Define keywords and corresponding regex patterns
keywords_patterns = {
    'tv': r'\b(TV|HDTV|Roku|chromecast)\b',
    'netflix': r'\b(Netflix|Amazon|hulu|disney+)\b',
    'pool': r'\b(Pool|Swimming pool)\b',
    'oven': r'\b(Oven|microwave)\b',
    'stove': r'\b(Stove|stove)\b',
    'wifi': r'\b(Wifi)\b',
    'toiletries': r'\b(Toiletries|Shampoo|Brush|Conditioner|soap|wash|shower gel)\b',
    'parking': r'\b(parking|carport)\b',
    'sound_system': r'\b(SoundSystem|Sound System|speakers|speaker|piano|record player)\b',
    'coffee_maker': r'\b(coffee|coffee-maker|coffeemaker|espresso)\b',
    'grill': r'\b(Grill|BBQ)\b',
    'workspace': r'\b(Workspace|Workspaces)\b',
    'closet': r'\b(Closet|storage|wardrobe|dresser)\b',
    'gym': r'\b(Exercise|yoga|elliptical|bike|bikes|gym)\b',
    'refrigerator': r'\b(Refrigerator|refrigerator|freezer|ice machine|fridge)\b',
    'housekeeping': r'\b(Housekeeping|Housekeeping)\b',
    'backyard': r'\b(backyard|patio|balcony)\b',
    'safety': r'\b(Safety|Safety|lock|pin|alarm|keypad)\b',
    'spa': r'\b(Sauna|sauna|jacuzzi|tub|spa|steam room|sun loungers)\b',
    'view': r'\b(lake|skyline|view|courtyard|resort|waterfront)\b',
    'laundry': r'\b(Laundry|laundry|laundromat)\b',
    'games': r'\b(Game console|gaming console|gaming consoles|ps2|ps3|ps4|ps5|xbox|nintendo|games|ping pong)\b',
    'first_aid': r'\b(first aid)\b',
    'smoke_alarm': r'\b(smoke alarm|fire extinguisher|alarm)\b',
    'private_entrance': r'\b(private entrance)\b',
    'AC': r'\b(heating|conditioning|AC)\b',
    'bedding': r'\b(pillows|pillow|blanket|blankets|bed linens)\b',
    'baby_ameneties': r'\b(crib|baby|high chair|changing table|playroom|playground)\b',
    'utensils': r'\b(utensils|cooking|dishes|silverware|glasses)\b',
    'kitchen': r'\b(kitchen|dining|rice maker|blender|kitchenette)\b',
    'bathtub': r'\b(bathtub)\b',
    'iron': r'\b(iron|ironing board)\b',
    'self_checking': r'\b(Self check-in|check-in|self_checking|self checkin|self checking)\b',
    'hair_dryer': r'\b(hair dryer)\b',
    'security_camera': r'\b(security|camera|cameras)\b',
    'toaster': r'\b(toaster|bread maker)\b',
    'pets_allowed' : r'\b(pets allowed)\b',
    'bookshelf': r'\b(bookshelf|books|reading)\b',
    'cleaning products': r'\b(cleaning products)\b',
    'fire pit': r'\b(fire pit|fire place|firepit|fireplace)\b',
    'garage': r'\b(garage)\b',
    'beach': r'\b(beach)\b',
    'host_there': r'\b(host greets you|property manager)\b',
    'bar': r'\b(bar)\b',
    'sports': r'\b(kayak|golf|ski|ski-in/ski-out|lasertag|laser tag|batting cage|wall climbing|climbing wall|bowling|hockey rink|skate ramp)\b',
    'ev_charger': r'\b(ev charger)\b',
    'movie_theater': r'\b(movie theater|media room|theme room)\b',
}
# Function to rename items based on regex patterns
def rename_item(amenity, patterns):
    for key, pattern in patterns.items():
        if re.search(pattern, amenity, flags=re.IGNORECASE):
            return key
    return None
# Function to rename items in the amenities list
def rename_amenities_list(amenities_list, patterns):
    return [rename_item(amenity, patterns) for amenity in amenities_list if rename_item(amenity, patterns) is not None]
# Filter out any empty strings from unique_amenities
filtered_unique_amenities = [amenity for amenity in unique_items if amenity.strip()]
# Apply the function to rename items in the filtered 'unique_amenities' list using list comprehension
updated_amenities = [rename_item(amenity, keywords_patterns) for amenity in filtered_unique_amenities]
# Remove items that did not match any keyword pattern
updated_amenities = [amenity for amenity in updated_amenities if amenity is not None]
# Extract unique items from the list in the column
unique_updated_amenities = list(set(updated_amenities))

# Correctly rename items in the 'amenities' column
data_train['renamed_amenities'] = data_train['amenities'].apply(lambda x:
    rename_amenities_list(safe_eval(x), keywords_patterns) if isinstance(x, str) else [])
# Apply one hot encoding to the unique_items and create a new column for each item
for item in unique_updated_amenities:
    column_name = item + "_amenity"
    data_train[column_name] = data_train['renamed_amenities'].apply(lambda x: item in x)
# Drop the original amenities and renamed_amenities columns
data_train.drop(['amenities', 'renamed_amenities'], axis=1, inplace=True)

# Sentiment Analysis and polarity scores of title and description
sia= SentimentIntensityAnalyzer()
data_train['title_scores'] = data_train['title'].apply(lambda title: sia.polarity_scores(str(title)))
data_train['title_sentiment']=data_train['title_scores'].apply(lambda score_dict:score_dict['compound'])
data_train.drop(['title', 'title_scores'], axis=1, inplace=True)
data_train['description_scores']=data_train['description'].apply(lambda description: sia.polarity_scores(str(description)))
data_train['description_sentiment']=data_train['description_scores'].apply(lambda score_dict:score_dict['compound'])
data_train.drop(['description', 'description_scores'], axis=1, inplace=True)

# Training dataset feature selection
train_features = data_train.drop(['id', 'neighbourhood'], axis=1)


### Replicating the changes to test dataset

# Converting values in price column to its natural logarithm
data_test['price'] = np.log(data_test['price'])

# Fill null value with unlicensed
data_test['license'].fillna('Unlicensed', inplace=True)

# Drop rows with empty price, and reviews
data_test = data_test.dropna(subset=['price'])
data_test = data_test.dropna(subset=['review_scores_rating'])
data_test = data_test.dropna(subset=['review_sentiment_score'])

# Fill null in host realted columns
data_test['host_is_superhost'] = data_test['host_is_superhost'].fillna('f')
data_test['has_availability'] = data_test['has_availability'].fillna('f')
data_test['host_response_time'] = data_test['host_response_time'].fillna(data_test['host_response_time'].mode()[0])

# Remove % sign and convert to numeric for the following columns
data_test['host_response_rate'] = pd.to_numeric(data_test['host_response_rate'].str.replace('%', ''))
data_test['host_acceptance_rate'] = pd.to_numeric(data_test['host_acceptance_rate'].str.replace('%', ''))

# Fill null values in specified columns with median values
for column in numeric_columns:
    median_value = data_test[column].median()
    data_test[column].fillna(median_value, inplace=True)

# Adding days since columns
# Converting date columns
data_test['host_since'] = pd.to_datetime(data_test['host_since'])
# Calculating values and storing in a new column
data_test['host_since_days'] = (current_date - data_test['host_since']).dt.days
# Dropping date columns
data_test.drop(columns=['host_since', 'first_review_date', 'last_review_date'], inplace=True)

# Calculate distance to downtown using broadcasting
for city, (lat, lon) in downtown_coords.items():
    mask = data_test['city'] == city
    data_test.loc[mask, 'distance_to_downtown'] = haversine_distance(
        data_test.loc[mask, 'latitude'].values,
        data_test.loc[mask, 'longitude'].values,
        lat,
        lon
    )

# Calculate attractions within 10km for each Airbnb listing
attraction_counts = []
for city in data_test['city'].unique():
    attractions_city = attractions_df[attractions_df['city'] == city]
    if not attractions_city.empty:
        attraction_counts.extend(data_test[data_test['city'] == city].apply(
            lambda row: count_attractions_within_radius(row['latitude'], row['longitude'], attractions_city),
            axis=1
        ))
    else:
        attraction_counts.extend([0] * len(data_test[data_test['city'] == city]))
data_test['nearby_attractions'] = attraction_counts

# Converting boolean columns and picture url columns to 0s and 1s
for boolean_column in ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
    data_test[boolean_column] = data_test[boolean_column].map(lambda s: False if s == "f" else True)

# Changing data in license column to licensed and unlicensed and converting to boolean
data_test['license'] = data_test['license'].map(lambda s: False if s == "Unlicensed" else True)

# Ordinal Encoding host_response_time since there is a clear order
data_test['host_response_time_encoded'] = ordinal_encoder.fit_transform(data_test[['host_response_time']])
data_test.drop(columns=['host_response_time'], inplace=True)

# One hot encoding property_type, city, and room type
dummy_vars = pd.get_dummies(data_test['property_type'], prefix='property')
data_test = pd.concat([data_test, dummy_vars], axis=1)
dummy_vars = pd.get_dummies(data_test['city'], prefix='city')
data_test = pd.concat([data_test, dummy_vars], axis=1)
dummy_vars = pd.get_dummies(data_test['room_type'], prefix='room_type')
data_test = pd.concat([data_test, dummy_vars], axis=1)

# Extracting and One hot encoding verifications
unique_items = set(item for sublist in data_test['host_verifications'].apply(safe_eval) for item in sublist)
for item in unique_items:
    column_name = item + "_verification"
    data_test[column_name] = data_test['host_verifications'].apply(lambda x: True if item in x else False)
data_test.drop('host_verifications', axis=1, inplace=True)

# Extracting and one hot encoding amenities
unique_items = set(item for sublist in data_test['amenities'].apply(safe_eval) for item in sublist if item)
filtered_unique_amenities = [amenity for amenity in unique_items if amenity.strip()]
updated_amenities = [rename_item(amenity, keywords_patterns) for amenity in filtered_unique_amenities]
updated_amenities = [amenity for amenity in updated_amenities if amenity is not None]
unique_updated_amenities = list(set(updated_amenities))
data_test['renamed_amenities'] = data_test['amenities'].apply(lambda x: rename_amenities_list(safe_eval(x),
    keywords_patterns) if isinstance(x, str) else [])
for item in unique_updated_amenities:
    column_name = item + "_amenity"
    data_test[column_name] = data_test['renamed_amenities'].apply(lambda x: item in x)
data_test.drop(['amenities', 'renamed_amenities'], axis=1, inplace=True)

# Calculating title and description sentiment scores
data_test['title_scores'] = data_test['title'].apply(lambda title: sia.polarity_scores(str(title)))
data_test['title_sentiment']=data_test['title_scores'].apply(lambda score_dict:score_dict['compound'])
data_test.drop(['title', 'title_scores'], axis=1, inplace=True)
data_test['description_scores']=data_test['description'].apply(lambda description: sia.polarity_scores(str(description)))
data_test['description_sentiment']=data_test['description_scores'].apply(lambda score_dict:score_dict['compound'])
data_test.drop(['description', 'description_scores'], axis=1, inplace=True)

# Testing dataset feature selection
test_features = data_test.drop(['id', 'neighbourhood'], axis=1)


################ DATA PREPARATION ################

# Identifying and dropping encoded columns for tuned LightGBM and separate features
cols_to_drop = train_features.filter(regex='^(property_(?!type$)|city_|room_type_)').columns
X_train_selected = train_features.drop(columns=cols_to_drop)
X_train_selected = X_train_selected.drop(['price'], axis=1)
cols_to_drop = test_features.filter(regex='^(property_(?!type$)|city_|room_type_)').columns
X_test_selected = test_features.drop(columns=cols_to_drop)
X_test_selected = X_test_selected.drop(['price'], axis=1)

# Dropping categorical columns for all models pre-tuning
X_train = train_features.drop(columns=['price', 'property_type', 'city', 'room_type'])
X_test = test_features.drop(columns=['price', 'property_type', 'city', 'room_type'])

# Aligning column discrepancy due to separating property types. Important step.
common_columns = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_columns]
X_test = X_test[common_columns]

# Selecting target for train and test
y_train = train_features['price']
y_test = test_features['price']


################ MODEL BUILDING ################

### Linear Regression

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using PCA and skree plot to identify ideal number of components to be used in the model
pca = PCA()
pca.fit(X_train_scaled)
# Plot cumulative explained variance ratio
cumulative_variance = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(10, 10))
plt.plot(range(1, X_train_scaled.shape[1] + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance of PCA Components')
plt.grid(True)
plt.show()

# Recucing dimension using PCA
pca = PCA(n_components=98) # Using Cumulative Explained Variance of 95%
train_pca = pca.fit_transform(X_train_scaled)
test_pca = pca.transform(X_test_scaled)

# Building and evaluating the model
model_linear = LinearRegression()
model_linear.fit(train_pca, y_train)
# Predicting the price
train_pred_linear = model_linear.predict(train_pca)
test_pred_linear = model_linear.predict(test_pca)

l_train_r2 = r2_score(y_train, train_pred_linear)
l_train_mse = mean_squared_error(y_train, train_pred_linear)
l_test_r2 = r2_score(y_test, test_pred_linear)
l_test_mse = mean_squared_error(y_test, test_pred_linear)

print(f'\n--------Linear Regression Train Fitting-----------')
print(f'R2 Score: {l_train_r2}')
print(f'MSE: {l_train_mse}')
print(f'--------Linear Regression Test Fitting-----------')
print(f'R2 Score: {l_test_r2}')
print(f'MSE: {l_test_mse}')


### Ridge Regression

# Building and evaluating the model
model_ridge = Ridge(alpha=0.91, random_state=42) # Used RidgeCV to identify ideal hyperparameter
model_ridge.fit(X_train, y_train)

train_pred_ridge = model_ridge.predict(X_train)
test_pred_ridge = model_ridge.predict(X_test)

r_train_r2 = r2_score(y_train, train_pred_ridge)
r_train_mse = mean_squared_error(y_train, train_pred_ridge)
r_test_r2 = r2_score(y_test, test_pred_ridge)
r_test_mse = mean_squared_error(y_test, test_pred_ridge)

print(f'\n--------Ridge Regression Train Fitting-----------')
print(f'R2 Score: {r_train_r2}')
print(f'MSE: {r_train_mse}')
print(f'--------Ridge Regression Test Fitting-----------')
print(f'R2 Score: {r_test_r2}')
print(f'MSE: {r_test_mse}')


### Light GBM

# Building and evaluating the model
final_model_lgbm = lgb.LGBMRegressor(
    learning_rate=0.01, n_estimators=3000, num_leaves=200, verbose=0, max_depth=-1, random_state=42
)
final_model_lgbm.fit(X_train, y_train)

test_pred_flgbm = final_model_lgbm.predict(X_test)
train_pred_flgbm = final_model_lgbm.predict(X_train)

flgbm_train_r2 = r2_score(y_train, train_pred_flgbm)
flgbm_train_mse = mean_squared_error(y_train, train_pred_flgbm)
flgbm_test_r2 = r2_score(y_test, test_pred_flgbm)
flgbm_test_mse = mean_squared_error(y_test, test_pred_flgbm)

print(f'\n--------Light GBM Train Fitting-----------')
print(f'R2 Score: {flgbm_train_r2}')
print(f'MSE: {flgbm_train_mse}')
print(f'--------Light GBM Test Fitting-----------')
print(f'R2 Score: {flgbm_test_r2}')
print(f'MSE: {flgbm_test_mse}')


### XG Boost

# Building and evaluating the model
model_xgb = xgb.XGBRegressor(learning_rate=0.04, max_depth=10, min_child_weight=2, n_estimators=1500, seed=25)
model_xgb.fit(X_train, y_train)

train_pred_xgb = model_xgb.predict(X_train)
test_pred_xgb = model_xgb.predict(X_test)

xgb_train_r2 = r2_score(y_train, train_pred_xgb)
xgb_train_mse = mean_squared_error(y_train, train_pred_xgb)
xgb_test_r2 = r2_score(y_test, test_pred_xgb)
xgb_test_mse = mean_squared_error(y_test, test_pred_xgb)

print(f'\n--------XG Boost Train Fitting-----------')
print(f'R2 Score: {xgb_train_r2}')
print(f'MSE: {xgb_train_mse}')
print(f'--------XG Boost Test Fitting-----------')
print(f'R2 Score: {xgb_test_r2}')
print(f'MSE: {xgb_test_mse}')


### Random Forest Regressor

# Building and evaluating the model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

train_pred_rf = model_rf.predict(X_train)
test_pred_rf = model_rf.predict(X_test)

rf_train_r2 = r2_score(y_test, train_pred_rf)
rf_train_mse = mean_squared_error(y_test, train_pred_rf)
rf_test_r2 = r2_score(y_test, test_pred_rf)
rf_test_mse = mean_squared_error(y_test, test_pred_rf)

print(f'\n--------Random Forest Train Fitting-----------')
print(f'R2 Score: {rf_train_r2}')
print(f'MSE: {rf_train_mse}')
print(f'--------Random Forest Test Fitting-----------')
print(f'R2 Score: {rf_test_r2}')
print(f'MSE: {rf_test_mse}')


### Compare 3 models and select the best one
# Create a DataFrame for easy plotting
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'LightGBM', 'XGBoost', 'RF Regressor'],
    'R² Score': [l_test_r2, r_test_r2, flgbm_test_r2, xgb_test_r2, rf_test_r2],
    'MSE': [l_test_mse, r_test_mse, flgbm_test_mse, xgb_test_mse, rf_test_mse]
})
# Find the index of the highest R² score and lowest MSE
best_r2_idx = results_df['R² Score'].idxmax()
best_mse_idx = results_df['MSE'].idxmin()
# Set up the subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# Plot R² scores with custom colors
palette_r2 = ['red' if i == best_r2_idx else 'blue' for i in range(len(results_df))]
sns.barplot(ax=axes[0], x='Model', y='R² Score', data=results_df, palette=palette_r2)
axes[0].set_title('R² Score Comparison of Regression Models')
axes[0].set_xlabel('')
axes[0].set_ylabel('R² Score')
# Plot MSE with custom colors
palette_mse = ['red' if i == best_mse_idx else 'blue' for i in range(len(results_df))]
sns.barplot(ax=axes[1], x='Model', y='MSE', data=results_df, palette=palette_mse)
axes[1].set_title('Mean Squared Error Comparison of Regression Models')
axes[1].set_xlabel('')
axes[1].set_ylabel('Mean Squared Error')
plt.tight_layout()
plt.show()



################ MODEL TUNING ################

# Stacking features and target for interpretability
test = [X_test, y_test]

# Creating a function to calculate SHAP values and display visualization based on provided parameters
def XAI_SHAP(model, data, graph, obs=0):
    """ Computes SHAP values and represents XAI graphs
    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - graph = Global or local interpretation
        - obs = Index of data instance to explain
    - Output:
        - XAI graphs and SHAP values
    """
    # Print JavaScript visualizations
    shap.initjs()
    # Create object to calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    if graph == 'global':
        # Global Interpretability (feature importance)
        shap.plots.bar(shap_values, max_display=20)
        # Global Interpretability (impact on target variable)
        shap.summary_plot(shap_values, data, max_display=20)
    else:
        # Local Interpretability (coefficients)
        shap.plots.waterfall(shap_values[obs], max_display=20)
    return shap_values

# SHAP values and waterfall plots for Local Interpretability of predictions with highest residuals (calculated)
shap_values = XAI_SHAP(final_model_lgbm, test[0], 'local', 24)
shap_values = XAI_SHAP(final_model_lgbm, test[0], 'local', 165)
shap_values = XAI_SHAP(final_model_lgbm, test[0], 'local', 124)


################## Model Interpretation and Testing Findings and Chnages #################
##### Changed how amenities are extracted - reduced overfitting
##### Removed more features based on domain knowledge - improved accuracy </li>
##### Used categorical features directly - better model interpretability </li>
##### Global and local interpretetions showed that last_review_days, first_review_days, number_of_reviews,
# and host_listings_count was having a negative effect on price prediction, so we removed that column for better accuracy. </li>
##### Identified the predictions with highest residuals and interpreted them using SHAP -  Results indicate the
# model's pricing/recommended price is logical, and the original listing price is too high for it's features.
##### Increased train size - reduced overfitting
##### Improved parameters - reduced overfitting


### Building the tuned model

# Reduced Number of estimators to reduce number of iterations of boosting
# Limited number of leaves to 50
# Limited maximum number of bins
model_lgbm = lgb.LGBMRegressor(
    max_bin=100, learning_rate=0.01, n_estimators=1000, num_leaves=50, verbose=0, max_depth=-1, random_state=42
)

# Encoding categorical features to be used directly in LGBM
categorical_features = ['property_type', 'city', 'room_type']
for col in categorical_features:
    X_train_selected[col] = X_train_selected[col].astype('category')
    X_test_selected[col] = X_test_selected[col].astype('category')

# Converting categorical features to their indices
categorical_feature_indices = [X_train_selected.columns.get_loc(col) for col in categorical_features]

# Building and evaluating the model
model_lgbm.fit(X_train_selected, y_train, categorical_feature=categorical_feature_indices)
train_pred_lgbm = model_lgbm.predict(X_train_selected)
test_pred_lgbm = model_lgbm.predict(X_test_selected)

lgbm_train_r2 = r2_score(y_train, train_pred_lgbm)
lgbm_train_mse = mean_squared_error(y_train, train_pred_lgbm)
lgbm_test_r2 = r2_score(y_test, test_pred_lgbm)
lgbm_test_mse = mean_squared_error(y_test, test_pred_lgbm)

print(f'\n--------Tuned Light GBM Train Fitting-----------')
print(f'R2 Score: {lgbm_train_r2}')
print(f'MSE: {lgbm_train_mse}')
print(f'--------Tuned Light GBM Test Fitting-----------')
print(f'R2 Score: {lgbm_test_r2}')
print(f'MSE: {lgbm_test_mse}')

r2_difference = lgbm_train_r2 - lgbm_test_r2
mse_difference = lgbm_train_mse - lgbm_test_mse
print(f'Difference in R2 Score: {r2_difference}')
print(f'Difference in MSE: {mse_difference}')


################ MODEL SYSTEM DEVELOPMENT ################

# Building a cluster of similar listings nearby and displaying at the end
# Selecting necessary features
cluster_df = airbnb_df[[
    'id', 'title', 'property_type', 'room_type', 'latitude', 'longitude', 'accommodates', 'minimum_nights', 'amenities', 'price'
]]
cluster_df = cluster_df.dropna(subset=['price'])

# Create a LabelEncoder instance
label_encoder_property_type = LabelEncoder()
label_encoder_room_type = LabelEncoder()
# Apply label encoding to 'property_type' and 'room_type'
cluster_df['property_type_encoded'] = label_encoder_property_type.fit_transform(cluster_df['property_type'])
cluster_df['room_type_encoded'] = label_encoder_room_type.fit_transform(cluster_df['room_type'])
# Feature Engineering
features = cluster_df[['accommodates', 'room_type_encoded', 'property_type_encoded']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Determine the ideal number of clusters using the elbow plot
wcss = []
max_clusters = 20
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)
# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-cluster sum of squares)')
plt.title('Elbow Plot to Determine Optimal Number of Clusters')
plt.show()

# Clustering using KMeans
# Using high cluster number because we want to select very distinct listings
kmeans = KMeans(n_clusters=15, random_state=42)
cluster_df['cluster'] = kmeans.fit_predict(features_scaled)


#----------------------------------#
# Example usage: find similar listings for a randomly selected target listing
target_listing_id = np.random.choice(cluster_df['id'])
# Find the cluster of the target listing
target_cluster = cluster_df[cluster_df['id'] == target_listing_id]['cluster'].values[0]

# Get the latitude and longitude of the target listing
target_location = cluster_df[cluster_df['id'] == target_listing_id][['latitude', 'longitude']].values[0]

# Filter the listings that fall into the same cluster
same_cluster_listings = cluster_df[cluster_df['cluster'] == target_cluster]

# Function to check if a listing is within a given radius
def is_within_radius(row, target_location, radius_km):
    listing_location = (row['latitude'], row['longitude'])
    return great_circle(listing_location, target_location).km <= radius_km

# Apply the distance function
same_cluster_listings['within_radius'] = same_cluster_listings.apply(is_within_radius, target_location=target_location, radius_km=2, axis=1)

# Filter listings that are within the 2km radius
within_radius_listings = same_cluster_listings[same_cluster_listings['within_radius']]

# Drop the 'within_radius' column as it's no longer needed
within_radius_listings = within_radius_listings.drop(columns=['within_radius'])

# Exclude the target listing from the final result
final_result = within_radius_listings[within_radius_listings['id'] != target_listing_id]

print(f'{within_radius_listings.shape[0]} similar listings found nearby.')

# Decode the label encoding for 'property_type' and 'room_type'
within_radius_listings['property_type'] = label_encoder_property_type.inverse_transform(
    within_radius_listings['property_type_encoded'])
within_radius_listings['room_type'] = label_encoder_room_type.inverse_transform(
    within_radius_listings['room_type_encoded'])

# Display specified columns
display_columns = ['title', 'property_type', 'room_type', 'accommodates', 'minimum_nights', 'amenities', 'latitude', 'longitude', 'price']
result = within_radius_listings[display_columns]

# Display the result
result.head()