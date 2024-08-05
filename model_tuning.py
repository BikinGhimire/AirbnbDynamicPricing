import argparse
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from scipy import stats
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import re
# import nltk
# nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from numpy import logspace
import lightgbm as lgb
import xgboost as xgb
from sklearn import ensemble
import warnings
warnings.filterwarnings('ignore')


def load_data():
    cities = ['Montreal', 'NewBrunswick', 'Ottawa', 'QuebecCity', 'Toronto', 'Vancouver', 'Victoria', 'Winnipeg']
    dataframes = []
    
    for city in cities:
        df = pd.read_csv(f'Dataset/Airbnb/{city}.csv')
        df['city'] = city.replace('NewBrunswick', 'New Brunswick').replace('QuebecCity', 'Quebec City')
        dataframes.append(df)
    
    airbnb_df = pd.concat(dataframes, ignore_index=True)
    reviews_df = pd.read_csv('Dataset/Sentiment/listing_sentiment_scores.csv')
    airbnb_df = pd.merge(airbnb_df, reviews_df, left_on='id', right_on='listing_id', how='left')
    
    return airbnb_df

def clean_data(df):
    feature_elimination_list = ['listing_url', 'scrape_id', 'last_scraped', 'source', 'host_url', 'host_thumbnail_url', 'calendar_updated', 'calendar_last_scraped', 'host_name', 'listing_id', 'host_id', 'neighbourhood_group_cleansed', 'host_neighbourhood', 'host_location', 'neighbourhood', 'host_listings_count', 'host_total_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'reviews_per_month', 'review_scores_accuracy', 'bathrooms_text', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'host_about', 'neighborhood_overview', 'host_has_profile_pic', 'picture_url', 'host_picture_url']
    df = df.drop(feature_elimination_list, axis=1)
    
    column_mappings = {
        'name': 'title',
        'first_review': 'first_review_date',
        'last_review': 'last_review_date',
        'review_scores_value': 'review_scores_value_for_money',
        'calculated_host_listings_count': 'host_listings_count',
        'neighbourhood_cleansed': 'neighbourhood',
        'sentiment_score': 'review_sentiment_score'
    }
    df.rename(columns=column_mappings, inplace=True)
    
    df['price'] = df['price'].str.replace('[$,]', '', regex=True).astype(float)
    # df['license'].fillna('Unlicensed', inplace=True)
    # df = df.dropna(subset=['price', 'number_of_reviews', 'review_sentiment_score'])
    
    # today_date = datetime.today().strftime('%Y-%m-%d')
    # df.loc[df['number_of_reviews'] == 0, 'first_review_date'] = df.loc[df['number_of_reviews'] == 0, 'first_review_date'].fillna(today_date)
    # df.loc[df['number_of_reviews'] == 0, 'last_review_date'] = df.loc[df['number_of_reviews'] == 0, 'last_review_date'].fillna(today_date)
    
    # df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
    # df['has_availability'] = df['has_availability'].fillna('f')
    # df['host_response_time'] = df['host_response_time'].fillna(df['host_response_time'].mode()[0])
    
    # df['host_response_rate'] = pd.to_numeric(df['host_response_rate'].str.replace('%', ''))
    # df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'].str.replace('%', ''))
    
    # numeric_columns = ['host_response_rate', 'host_acceptance_rate', 'bedrooms', 'beds', 'review_scores_value_for_money', 'review_scores_location', 'review_scores_checkin', 'review_scores_communication', 'review_scores_cleanliness', 'bathrooms']
    # for column in numeric_columns:
    #     df[column].fillna(df[column].median(), inplace=True)
    
    return df

def preprocess_data(df):
    # Converting values in price column to its natural logarithm
    df['price'] = np.log(df['price'])
    
    # Fill null value with 'Unlicensed'
    df['license'].fillna('Unlicensed', inplace=True)
    
    # Drop rows with empty price
    df = df.dropna(subset=['price'])
    
    # Drop rows with empty values in specified columns
    df = df.dropna(subset=['number_of_reviews', 'review_sentiment_score'])
    
    # Get today's date
    today_date = datetime.today().strftime('%Y-%m-%d')
    
    # Fill null values with today's date where 'number_of_reviews' equals 0
    df.loc[df['number_of_reviews'] == 0, 'first_review_date'] = df.loc[df['number_of_reviews'] == 0, 'first_review_date'].fillna(today_date)
    df.loc[df['number_of_reviews'] == 0, 'last_review_date'] = df.loc[df['number_of_reviews'] == 0, 'last_review_date'].fillna(today_date)
    
    df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
    df['has_availability'] = df['has_availability'].fillna('f')
    df['host_response_time'] = df['host_response_time'].fillna(df['host_response_time'].mode()[0])
    
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'].str.replace('%', ''))
    df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'].str.replace('%', ''))
    
    # Fill null values in specified columns with median values
    numeric_columns = ['host_response_rate', 'host_acceptance_rate', 'bedrooms', 'beds', 'review_scores_value_for_money',
                       'review_scores_location', 'review_scores_checkin', 'review_scores_communication', 'review_scores_cleanliness', 'bathrooms']
    for column in numeric_columns:
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)
    
    # Feature Engineering
    current_date = datetime.now()
    
    # Converting date columns
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['first_review_date'] = pd.to_datetime(df['first_review_date'])
    df['last_review_date'] = pd.to_datetime(df['last_review_date'])
    
    # Calculating values and storing in a new column
    df['host_since_days'] = (current_date - df['host_since']).dt.days
    df['first_review_days'] = (current_date - df['first_review_date']).dt.days
    df['last_review_days'] = (current_date - df['last_review_date']).dt.days
    
    # Dropping date columns
    df.drop(columns=['host_since', 'first_review_date', 'last_review_date'], inplace=True)
    
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
        mask = df['city'] == city
        df.loc[mask, 'distance_to_downtown'] = haversine_distance(
            df.loc[mask, 'latitude'].values,
            df.loc[mask, 'longitude'].values,
            lat, lon
        )
    
    # Load attraction data
    attractions_df = pd.read_csv('Dataset/Locations/canadian_tourist_attractions.csv')
    attractions_df['city'] = attractions_df['city'].str.title().str.replace('_', ' ')
    
    def count_attractions_within_radius(airbnb_lat, airbnb_lon, attractions, radius_km=10):
        distances = haversine_distance(airbnb_lat, airbnb_lon, attractions['latitude'].values, attractions['longitude'].values)
        return np.sum(distances <= radius_km)
    
    # Calculate attractions within 10km for each Airbnb listing
    attraction_counts = []
    for city in df['city'].unique():
        attractions_city = attractions_df[attractions_df['city'] == city]
        if not attractions_city.empty:
            attraction_counts.extend(df[df['city'] == city].apply(
                lambda row: count_attractions_within_radius(row['latitude'], row['longitude'], attractions_city),
                axis=1
            ))
        else:
            attraction_counts.extend([0] * len(df[df['city'] == city]))
    
    df['nearby_attractions'] = attraction_counts
    
    # Data Encoding
    for boolean_column in ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
        df[boolean_column] = df[boolean_column].map(lambda s: False if s == "f" else True)
    
    df['license'] = df['license'].map(lambda s: False if s == "Unlicensed" else True)
    
    # Define the order of categories
    categories = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']
    
    # Initialize OrdinalEncoder with the defined categories
    ordinal_encoder = OrdinalEncoder(categories=[categories])
    
    # Fit and transform the 'host_response_time' column
    df['host_response_time_encoded'] = ordinal_encoder.fit_transform(df[['host_response_time']])
    
    # Dropping the categorical column
    df.drop(columns=['host_response_time'], inplace=True)
    
    # Function to safely evaluate strings
    def safe_eval(x):
        if isinstance(x, str):
            return eval(x)
        return x
    
    # Extract unique items from the list in the column
    unique_items = set(item for sublist in df['host_verifications'].apply(safe_eval) for item in sublist)
    
    # Apply one hot encoding to the unique_items and create a new column for each item
    for item in unique_items:
        column_name = item + "_verification"
        df[column_name] = df['host_verifications'].apply(lambda x: True if item in x else False)
    
    # Drop the original amenities column
    df.drop('host_verifications', axis=1, inplace=True)
    
    # Feature Extraction
    unique_items = set(item for sublist in df['amenities'].apply(safe_eval) for item in sublist if item)
    
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
        'baby_amenities': r'\b(crib|baby|high chair|changing table|playroom|playground)\b',
        'utensils': r'\b(utensils|cooking|dishes|silverware|glasses)\b',
        'kitchen': r'\b(kitchen|dining|rice maker|blender|kitchenette)\b',
        'bathtub': r'\b(bathtub)\b'
    }
    
    def rename_item(amenity, patterns):
        for key, pattern in patterns.items():
            if re.search(pattern, amenity, flags=re.IGNORECASE):
                return key
        return None
    
    def rename_amenities_list(amenities_list, patterns):
        return [rename_item(amenity, patterns) for amenity in amenities_list if rename_item(amenity, patterns) is not None]
    
    filtered_unique_amenities = [amenity for amenity in unique_items if amenity.strip()]
    updated_amenities = [rename_item(amenity, keywords_patterns) for amenity in filtered_unique_amenities]
    updated_amenities = [amenity for amenity in updated_amenities if amenity is not None]
    unique_updated_amenities = list(set(updated_amenities))

    df['renamed_amenities'] = df['amenities'].apply(lambda x: rename_amenities_list(safe_eval(x), keywords_patterns) if isinstance(x, str) else [])

    # Apply one hot encoding to the unique_items and create a new column for each item
    for item in unique_updated_amenities:
        column_name = item + "_amenity"
        df[column_name] = df['renamed_amenities'].apply(lambda x: item in x)

    # Drop the original amenities and renamed_amenities columns
    df.drop(['amenities', 'renamed_amenities'], axis=1, inplace=True)

    sia= SentimentIntensityAnalyzer()

    # creating new columns using polarity scores function
    df['title_scores'] = df['title'].apply(lambda title: sia.polarity_scores(str(title)))
    df['title_sentiment']=df['title_scores'].apply(lambda score_dict:score_dict['compound'])
    df.drop(['title', 'title_scores'], axis=1, inplace=True)


    # creating new columns using polarity scores function
    df['description_scores']=df['description'].apply(lambda description: sia.polarity_scores(str(description)))
    df['description_sentiment']=df['description_scores'].apply(lambda score_dict:score_dict['compound'])
    df.drop(['description', 'description_scores'], axis=1, inplace=True)    
    
    return df

def convert_to_categorical(df, categorical_features):
    for feature in categorical_features:
        if feature in df.columns:
            if df[feature].dtype == 'object':
                df[feature] = df[feature].astype('category')
            elif df[feature].dtype.name != 'category':
                df[feature] = df[feature].astype('category')
    return df


def get_categorical_features(df):
    return [col for col in df.columns if df[col].dtype.name in ['category', 'object']]


def split_data(df):
    X = df.drop('price', axis=1)
    y = df['price']
    return train_test_split(X, y, test_size=0.01, random_state=42)

def train_model(X_train, y_train,categorical_features):
    # Encode categorical features
    categorical_features = ['property_type', 'city', 'room_type']
    for col in categorical_features:
        X_train[col] = X_train[col].astype('category')
        
        
    # Convert categorical features to their indices
    categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features]
    
    final_model_lgbm = lgb.LGBMRegressor(max_bin=100, learning_rate=0.01, n_estimators=1000, num_leaves=50, verbose=0, max_depth=-1, random_state=42)

    # X_train = convert_to_categorical(X_train, categorical_features)
    final_model_lgbm.fit(X_train, y_train, categorical_feature=categorical_feature_indices)
    return final_model_lgbm

def save_model_and_metadata(model, X_train):
    with open('airbnb_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()
    metadata = {
        'feature_names': X_train.columns.tolist(),
        'categorical_columns': categorical_features
    }
    with open('model_metadata.pkl', 'wb') as file:
        pickle.dump(metadata, file)


def load_model_and_metadata():
    with open('airbnb_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_metadata.pkl', 'rb') as file:
        metadata = pickle.load(file)
    return model, metadata

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse

def interpret_model(model, X):
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    return feature_importance_df


def predict_price(features, model, metadata):
    feature_names = metadata['feature_names']
    categorical_columns = metadata['categorical_columns']
    
    input_df = pd.DataFrame([features])
    
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Convert categorical columns to category dtype
    for col in categorical_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')
    
    # Ensure numeric columns are of float type
    numeric_columns = [col for col in feature_names if col not in categorical_columns]
    for col in numeric_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(float)
    
    input_df = input_df[feature_names]
    
    prediction = model.predict(input_df)
    return np.exp(prediction[0])





def main(args):
    if args.load_data or args.run_all:
        airbnb_df = load_data()
        print("Data loaded successfully")

    if args.clean_data or args.run_all:
        if 'airbnb_df' not in locals():
            airbnb_df = load_data()
        airbnb_df = clean_data(airbnb_df)
        print("Data cleaned successfully")

    if args.preprocess or args.run_all:
        if 'airbnb_df' not in locals():
            airbnb_df = load_data()
            airbnb_df = clean_data(airbnb_df)
        
        # Split the data first
        data_train, data_test = train_test_split(airbnb_df, test_size=0.01, random_state=42)
        
        # Preprocess the training data
        data_train = preprocess_data(data_train)
        print("Data preprocessed successfully")

    if args.train_model or args.run_all:
        if 'data_train' not in locals() or 'data_test' not in locals():
            if 'airbnb_df' not in locals():
                airbnb_df = load_data()
                airbnb_df = clean_data(airbnb_df)
            data_train, data_test = train_test_split(airbnb_df, test_size=0.01, random_state=42)
            data_train = preprocess_data(data_train)
        
        X_train = data_train.drop('price', axis=1)
        y_train = data_train['price']
        
        categorical_features = get_categorical_features(X_train)
        X_train = convert_to_categorical(X_train, categorical_features)
        
        model = train_model(X_train, y_train, categorical_features)
        save_model_and_metadata(model, X_train)
        print("Model trained and saved successfully")

    if args.evaluate_model or args.run_all:
        if 'model' not in locals():
            model, metadata = load_model_and_metadata()
        
        if 'data_test' not in locals():
            if 'airbnb_df' not in locals():
                airbnb_df = load_data()
                airbnb_df = clean_data(airbnb_df)
            _, data_test = train_test_split(airbnb_df, test_size=0.01, random_state=42)
        
        data_test = preprocess_data(data_test)
        X_test = data_test[metadata['feature_names']]
        y_test = data_test['price']

        categorical_features = get_categorical_features(X_test)
        X_test = convert_to_categorical(X_test, categorical_features)
        
        r2, mae, mse = evaluate_model(model, X_test, y_test)
        print(f"Model Evaluation Results:\nR2 Score: {r2}\nMAE: {mae}\nMSE: {mse}")

    if args.interpret_model or args.run_all:
        if 'model' not in locals():
            model, metadata = load_model_and_metadata()
            X_train = X_train[metadata['feature_names']]
        
        feature_importance_df = interpret_model(model, X_train)
        print("Top 10 Most Important Features:")
        print(feature_importance_df.head(10))

    if args.predict:
        model, metadata = load_model_and_metadata()
        features = {}
        for feature in args.predict:
            key, value = feature.split('=')
            features[key] = value
        
        # Convert boolean features
        boolean_features = ['has_availability', 'instant_bookable', 'host_is_superhost', 'host_identity_verified']
        for feature in boolean_features:
            if feature in features:
                features[feature] = features[feature].lower() == 'true'

        # Convert numeric features to appropriate types
        numeric_features = [col for col in metadata['feature_names'] if col not in metadata['categorical_columns']]
        for feature in numeric_features:
            if feature in features:
                features[feature] = float(features[feature])
        
                
        # Ensure categorical features are strings
        categorical_features = metadata['categorical_columns']
        for feature in categorical_features:
            if feature in features:
                features[feature] = str(features[feature])
        
        predicted_price = predict_price(features, model, metadata)
        print(f"Predicted price: ${predicted_price:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airbnb Price Prediction Model")
    parser.add_argument("--load_data", action="store_true", help="Load and merge datasets")
    parser.add_argument("--clean_data", action="store_true", help="Clean the data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train_model", action="store_true", help="Train the model")
    parser.add_argument("--evaluate_model", action="store_true", help="Evaluate the model")
    parser.add_argument("--interpret_model", action="store_true", help="Interpret the model")
    parser.add_argument("--run_all", action="store_true", help="Run all steps")
    parser.add_argument("--predict", nargs='+', help="Predict price for given features (e.g., city=Toronto accommodates=4)")

    args = parser.parse_args()
    main(args)