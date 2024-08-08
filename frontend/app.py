import pickle
import numpy as np
from math import exp
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import OrdinalEncoder
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
import folium
from nltk.data import find, path
import nltk

def ensure_nltk_resources():
    try:
        find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

# Set up header and brief description
with st.container():
    st.title('Airbnb Price Predictor')
    st.markdown('Provide data about your Airbnb listing and get predictions!')

# Begin new section for listings features
st.markdown('---')
st.subheader('Airbnb Listing characteristics')


city = st.selectbox('Select city', (
'Montreal', 'New Brunswick', 'Ottawa', 'Quebec City', 'Toronto', 'Vancouver', 'Victoria', 'Winnipeg'))


# Get the coordinates of the selected city
cities = {
    'Montreal': [45.5017, -73.5673],
    'New Brunswick': [45.2733, -66.0633],
    'Ottawa': [45.4215, -75.6972],
    'Quebec City': [46.8139, -71.2080],
    'Toronto': [43.7000, -79.4000],
    'Vancouver': [49.2827, -123.1207],
    'Victoria': [48.4284, -123.3656],
    'Winnipeg': [49.8951, -97.1384]
}

city_coords = cities[city]

# Create a Folium map centered on the selected city
m = folium.Map(location=city_coords, zoom_start=12)

# Add a draggable marker for the city center
marker = folium.Marker(
    location=city_coords,
    draggable=True,
    popup="Drag me to select a location"
)
marker.add_to(m)

# Display the map
st_data = st_folium(m, width=700, height=500)

# Get the selected location's latitude and longitude


# Get the marker's current position
if 'last_object_clicked' in st_data and st_data['last_object_clicked']:
    lat = st_data['last_object_clicked']['lat']
    lon = st_data['last_object_clicked']['lng']
    st.write(f"Marker Location: Latitude = {lat}, Longitude = {lon}")
else:
    lat = 0.0
    lon = 0.0
    st.write("Drag the marker to select a location.")

listing_title = st.text_input("Enter the list title")
listing_description = st.text_area("Enter the list description")

col1, col2 = st.columns(2)
with col1:
    accommodates = st.slider('Maximum Occupancy', 1, 16, 4)
    bathrooms = st.slider('Number of bathrooms', 1, 9, 2)
    room_type = st.selectbox('Room Type',
                             ('Private room', 'Entire apartment', 'Shared room', 'Hotel room'))
    listing_availability = st.selectbox('Is Listing Available?',('Yes','No'))
    instant_bookable = st.selectbox('Can the listing be instantly booked?',
                           ('No', 'Yes'))
    property_type =  st.selectbox('Select property type',('Entire rental unit', 'Entire home', 'Private room in rental unit',
       'Entire loft', 'Entire condo', 'Entire guest suite',
       'Private room in townhouse', 'Entire cottage',
       'Private room in condo', 'Private room in bed and breakfast',
       'Private room in home', 'Entire serviced apartment',
       'Entire townhouse', 'Private room', 'Room in aparthotel',
       'Private room in serviced apartment', 'Private room in loft',
       'Private room in cottage', 'Entire bungalow',
       'Entire vacation home', 'Shared room in rental unit', 'Tiny home',
       'Shared room in loft', 'Entire villa', 'Private room in bungalow',
       'Private room in guesthouse', 'Entire chalet',
       'Private room in hostel', 'Private room in villa',
       'Room in hostel', 'Entire guesthouse', 'Shared room in hostel',
       'Room in hotel', 'Shared room in home', 'Room in boutique hotel',
       'Private room in guest suite', 'Casa particular',
       'Shared room in condo', 'Private room in vacation home',
       'Private room in minsu', 'Religious building', 'Castle',
       'Entire cabin', 'Shared room in hotel', 'Boat', 'Entire place',
       'Campsite', 'Yurt', 'Camper/RV', 'Entire bed and breakfast',
       'Private room in farm stay', 'Barn', 'Room in bed and breakfast',
       'Private room in castle', 'Treehouse', 'Farm stay', 'Dome',
       'Room in serviced apartment', 'Private room in island',
       'Private room in tent', 'Private room in nature lodge',
       'Private room in earthen home', 'Private room in tiny home',
       'Private room in hut', 'Shared room in townhouse', 'Tent',
       'Island', 'Train', 'Private room in chalet',
       'Private room in casa particular', 'Private room in resort',
       'Shipping container', 'Tower', 'Bus', 'Private room in cabin',
       'Shared room in guesthouse', 'Cave', 'Shared room in tiny home',
       'Private room in cave', 'Entire home/apt', 'Private room in barn',
       'Floor', 'Shared room in bungalow', 'Shared room in guest suite',
       'Earthen home', 'Shared room in boat',
       'Shared room in bed and breakfast', 'Private room in treehouse',
       'Shared room', 'Private room in boat', 'Entire timeshare',
       'Private room in camper/rv', 'Houseboat', 'Private room in dome',
       'Private room in religious building'))

with col2:
    beds = st.slider('Number of beds', 1, 32, 2)
    bedrooms = st.slider('Number of bedrooms', 1, 24, 2)
    min_nights = st.slider('Minimum number of nights', 1, 20, 3)
    max_nights = st.slider('Maximum number of nights', 1, 20, 3)
    amenities = st.multiselect(
        'Select available amenities',
        ['sports', 'housekeeping', 'netflix', 'oven', 'AC', 'movie_theater', 'ev_charger', 'bathtub', 'toaster',
         'grill', 'fire_pit', 'kitchen', 'baby_ameneties',
         'host_there', 'toiletries', 'stove', 'bedding', 'workspace', 'garage', 'sound_system', 'games',
         'self_checking', 'parking', 'wifi', 'bookshelf', 'hair_dryer', 'laundry', 'spa', 'view', 'coffee_maker',
         'beach', 'safety', 'gym', 'utensils', 'backyard', 'cleaning_products', 'security_camera',
         'pets_allowed', 'tv', 'closet', 'iron', 'first_aid', 'smoke_alarm', 'refrigerator', 'pool',
         'private_entrance'],
        ['tv', 'wifi'])

number_of_days_available_in_a_month = st.slider('Number of days available', 1, 31, 1)
number_of_days_available_in_two_months = st.slider('Number of days available', 2, 60, 1)
number_of_days_available_in_three_months = st.slider('Number of days available', 3, 90, 1)
number_of_days_available_in_year = st.slider('Number of days available', 12, 365, 1)



# Section for host info
st.markdown('---')
st.subheader('Host Information')
col1, col2 = st.columns(2)
with col1:
    host_registration_date = st.date_input("Enter host registration date",datetime.today())
    # no_of_listings = st.slider('How many listings does the host have?', 1, 16, 4)
    super_host = st.selectbox('Is your host a superhost?', ('No', 'Yes'))
    host_response_rate = st.slider('Host Response Rate', 0, 100, 0)
    host_acceptance_rate = st.slider('Host Acceptance Rate', 0, 100, 0)
with col2:
    availability = st.selectbox('Is the listing available?', ('Yes', 'No'))
    response = st.selectbox('Response time', (
    'within an hour', 'within a few hours', 'within a day', 'a few days or more'))
    # no_of_reivews = st.slider('How many review does the host have?', 1, 16, 4)


host_license =st.selectbox('Is your host licensed?', ('Yes', 'No'))
host_identity_verified = st.selectbox('Is your host verified?', ('Yes', 'No'))
host_verifications =st.multiselect('Select available Host verifications',
                                   ['phone', 'email', 'work_email'],
                                   ['email']
                                   )




st.markdown('---')
st.subheader("Guests' feedback")
col1, col2, col3 = st.columns(3)


with col1:
    location = st.slider('Location rating', 1.0, 5.0, 4.0, step=0.5)
    checkin = st.slider('Checkin rating', 1.0, 5.0, 3.0, step=0.5)

with col2:
    clean = st.slider('Cleanliness rating', 1.0, 5.0, 3.0, step=0.5)
    communication = st.slider('Communication rating', 1.0, 5.0, 4.0, step=0.5)
with col3:
    value_for_money_rating = st.slider('Value for money rating', 1.0, 5.0, 3.5, step=0.5)
    overall_rating = st.slider('Overall rating', 1.0, 5.0, 4.0, step=0.5)

# first_review_date = st.date_input('First review date')
# last_review_date = st.date_input('Last review date')


text_review =st.text_area("Write your reviews here")




def inputdatapreprocess_encoding():

    global data, key, value, input_test, haversine_distance, item, sia


    # Create a dictionary to store the data of all the inputs including the location of the marker and dates as well
    data = {
        'city': city,
        'latitude': lat,
        'longitude': lon,
        'title': listing_title,
        'description': listing_description,
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': beds,
        'room_type': room_type,
        'instant_bookable': instant_bookable,
        'property_type': property_type,
        'minimum_nights': min_nights,
        'maximum_nights': max_nights,
        'amenities': [amenities],
        'availability_30': number_of_days_available_in_a_month,
        'availability_60': number_of_days_available_in_two_months,
        'availability_90': number_of_days_available_in_three_months,
        'availability_365': number_of_days_available_in_year,
        'host_since': host_registration_date,
        # 'host_listings_count': no_of_listings,
        'host_is_superhost': super_host,
        'host_response_rate': host_response_rate,
        'host_acceptance_rate': host_acceptance_rate,
        'has_availability': availability,
        'host_response_time': response,
        # 'number_of_reviews': no_of_reivews,
        'license': host_license,
        'host_identity_verified': host_identity_verified,
        'host_verifications': [host_verifications],
        'review_scores_location': location,
        'review_scores_checkin': checkin,
        'review_scores_cleanliness': clean,
        'review_scores_communication': communication,
        'review_scores_value_for_money': value_for_money_rating,
        'review_scores_rating': overall_rating,
        # 'first_review_date': first_review_date,
        # 'last_review_date': last_review_date,
        'text_review': text_review
    }


    # st.write(data)
    

    input_test = pd.DataFrame(data)

    # Display the DataFrame
    st.markdown('---')
    st.subheader('Summary of Inputs')
    st.write(input_test)

    # input_test = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    # print(input_test)
    input_test.dropna(inplace=True)
    print("First", input_test)
    today_date = datetime.today().strftime('%Y-%m-%d')
    # Converting date columns
    input_test['host_since'] = pd.to_datetime(input_test['host_since'])
    # input_test['first_review_date'] = pd.to_datetime(input_test['first_review_date'])
    # input_test['last_review_date'] = pd.to_datetime(input_test['last_review_date'])
    current_date = datetime.now()
    # Calculating values and storing in a new column
    input_test['host_since_days'] = (current_date - input_test['host_since']).dt.days
    # input_test['first_review_days'] = (current_date - input_test['first_review_date']).dt.days
    # input_test['last_review_days'] = (current_date - input_test['last_review_date']).dt.days
    # Dropping date columns
    input_test.drop(columns=['host_since'], inplace=True)
    # Number of attractions
    # attractions_df = pd.read_csv('Dataset/Locations/canadian_tourist_attractions.csv')
    attractions_df = pd.read_csv('./Dataset/Locations/canadian_tourist_attractions.csv')
    attractions_df['city'] = attractions_df['city'].str.title().str.replace('_', ' ')

    def count_attractions_within_radius(airbnb_lat, airbnb_lon, attractions, radius_km=10):
        distances = haversine_distance(airbnb_lat, airbnb_lon, attractions['latitude'].values,
                                       attractions['longitude'].values)
        return np.sum(distances <= radius_km)

    # # Define downtown coordinates for each city
    downtown_coords = {
        'Montreal': (45.5017, -73.5673),
        'New Brunswick': (45.9636, -66.6372),  # Fredericton
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
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def count_attractions_within_radius(lat, lon, attractions_df, radius=10):
        distances = attractions_df.apply(
            lambda row: haversine_distance(lat, lon, row['latitude'], row['longitude']),
            axis=1
        )
        return (distances <= radius).sum()

    # Selected_city
    selected_city = input_test['city'].values[0]
    # Get the coordinates of the selected city
    print(downtown_coords[selected_city])
    downtown_lat, downtown_log = downtown_coords[selected_city]
 
    mask = input_test['city'].values[0]
    downtown_lat, downtown_lon = downtown_coords[mask]
    input_test['distance_to_downtown'] = haversine_distance(
        input_test['latitude'].values[0],
        input_test['longitude'].values[0],
        downtown_lat,
        downtown_lon
    )
    # Calculate the number of attractions within 10km radius for the input location
    nearby_attractions_count = count_attractions_within_radius(input_test['latitude'].values[0],
                                                               input_test['longitude'].values[0], attractions_df)
    # Store the result in input_df
    input_test['nearby_attractions'] = nearby_attractions_count
    print(input_test)
    # st.write(attraction_counts)
    # Converting boolean columns and picture url columns to 0s and 1s
    for boolean_column in ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
        input_test[boolean_column] = input_test[boolean_column].map(lambda s: False if s == "f" else True)
    # Changing data in license column to licensed and unlicensed and converting to boolean
    input_test['license'] = input_test['license'].map(lambda s: False if s == "Unlicensed" else True)
    print(input_test['host_response_time'])
    # Initialize OrdinalEncoder with the defined categories
    categories = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']
    # Initialize OrdinalEncoder with the defined categories
    ordinal_encoder = OrdinalEncoder(categories=[categories])
    # Ordinal Encoding host_response_time since there is a clear order
    # Fit and transform the 'host_response_time' column
    input_test['host_response_time_encoded'] = ordinal_encoder.fit_transform(input_test[['host_response_time']])
    input_test.drop(columns=['host_response_time'], inplace=True)

    # print(unique_items)
    # Function to safely evaluate strings
    def safe_eval(x):
        if isinstance(x, str):
            return eval(x)
        return x

    # Extract unique items from the list in the column
    # unique_items = set(item for sublist in input_test['host_verifications'].apply(safe_eval) for item in sublist)
    unique_items = ["email", "phone", "work_email"]
    # Apply one hot encoding to the unique_items and create a new column for each item
    for item in unique_items:
        column_name = item + "_verification"
        input_test[column_name] = input_test['host_verifications'].apply(lambda x: True if item in x else False)
    # Drop the original amenities column
    input_test.drop('host_verifications', axis=1, inplace=True)
    # Amenities
    input_test['maximum_nights'] = max_nights
    input_test['minimum_nights'] = min_nights
    # unique_items =
    unique_items = ['sports', 'housekeeping', 'netflix', 'oven', 'AC', 'movie_theater', 'ev_charger', 'bathtub',
                    'toaster', 'grill', 'fire_pit', 'kitchen', 'baby_ameneties',
                    'host_there', 'toiletries', 'stove', 'bedding', 'workspace', 'garage', 'sound_system', 'games',
                    'self_checking', 'parking', 'wifi', 'bookshelf', 'hair_dryer', 'laundry', 'spa', 'view',
                    'coffee_maker',
                    'beach', 'safety', 'gym', 'utensils', 'backyard', 'cleaning_products', 'security_camera',
                    'pets_allowed', 'tv', 'closet', 'iron', 'first_aid', 'smoke_alarm', 'refrigerator', 'pool',
                    'private_entrance']
    print(len(unique_items))
    # Apply one hot encoding to the unique_items and create a new column for each item
    for item in unique_items:
        column_name = item + "_amenity"
        input_test[column_name] = input_test['amenities'].apply(lambda x: item in x)
    # Drop the original amenities and renamed_amenities columns
    input_test.drop(['amenities'], axis=1, inplace=True)
    # Sentiment Scores

    # creating an object of sentiment intensity analyzer
    ensure_nltk_resources()
    sia = SentimentIntensityAnalyzer()
    # creating new columns using polarity scores function
    input_test['title_scores'] = input_test['title'].apply(lambda title: sia.polarity_scores(str(title)))
    input_test['title_sentiment'] = input_test['title_scores'].apply(lambda score_dict: score_dict['compound'])
    input_test.drop(['title', 'title_scores'], axis=1, inplace=True)
    # creating new columns using polarity scores function
    input_test['description_scores'] = input_test['description'].apply(
        lambda description: sia.polarity_scores(str(description)))
    input_test['description_sentiment'] = input_test['description_scores'].apply(
        lambda score_dict: score_dict['compound'])
    input_test.drop(['description', 'description_scores'], axis=1, inplace=True)
    # creating a new column review_sentiment_score
    input_test['review_sentiment_scores'] = input_test['text_review'].apply(
        lambda review: sia.polarity_scores(str(review)))
    input_test['review_sentiment_score'] = input_test['review_sentiment_scores'].apply(
        lambda score_dict: score_dict['compound'])
    input_test.drop(['text_review', 'review_sentiment_scores'], axis=1, inplace=True)
    # Encode categorical features
    categorical_features = ['property_type', 'city', 'room_type']
    for col in categorical_features:
        input_test[col] = input_test[col].astype('category')
    # Convert categorical features to their indices
    categorical_feature_indices = [input_test.columns.get_loc(col) for col in categorical_features]

    # # Convert the DataFrame to a CSV file in memory
    csv = input_test.to_csv(index=False).encode('utf-8')

    # Display the DataFrame
    st.markdown('---')
    st.subheader('Data preprocessing')
    st.write(input_test)

    return input_test

input_test = inputdatapreprocess_encoding()


isPredictedPriceAvailable = False

# Price Prediction button
_, col2, _ = st.columns(3)
with col2:
    run_preds = st.button('Predict the price')
    if run_preds:
        # Load AI model from pickle file
        with open('./deployment/lgbm_model.pkl', 'rb') as f:
            lgbm_model = pickle.load(f)

        # Now you can safely use the test dataset for prediction
        # print(set(lgbm_model.feature_name_).difference(set(input_test.columns)))
        # print(set(input_test.columns).difference(set(lgbm_model.feature_name_)))

        input_test = input_test[sorted(input_test.columns)]
        predicted_price = lgbm_model.predict(input_test)

        # st.write(predicted_price)
        if predicted_price != 0:
            isPredictedPriceAvailable = True

        actual_predicted_price=int(np.exp(predicted_price))

        st.info(f"Predicted price is ${actual_predicted_price}")


if isPredictedPriceAvailable:
    st.subheader('Similar Listings Nearby')

    # Loading airbnb data of 8 cities
    montreal = pd.read_csv('./Dataset/Airbnb/Montreal.csv')
    newbrunswick = pd.read_csv('./Dataset/Airbnb/NewBrunswick.csv')
    ottawa = pd.read_csv('./Dataset/Airbnb/Ottawa.csv')
    quebeccity = pd.read_csv('./Dataset/Airbnb/QuebecCity.csv')
    toronto = pd.read_csv('./Dataset/Airbnb/Toronto.csv')
    vancouver = pd.read_csv('./Dataset/Airbnb/Vancouver.csv')
    victoria = pd.read_csv('./Dataset/Airbnb/Victoria.csv')
    winnipeg = pd.read_csv('./Dataset/Airbnb/Winnipeg.csv')

    # Adding a City column to each dataframes
    montreal['city'] = 'Montreal'
    newbrunswick['city'] = 'New Brunswick'
    ottawa['city'] = 'Ottawa'
    quebeccity['city'] = 'Quebec City'
    toronto['city'] = 'Toronto'
    vancouver['city'] = 'Vancouver'
    victoria['city'] = 'Victoria'
    winnipeg['city'] = 'Winnipeg'

    # Merging data from different cities to a single dataframe
    airbnb_df = pd.concat([montreal, newbrunswick, ottawa, quebeccity, toronto, vancouver, victoria, winnipeg],
                          ignore_index=True)

    cluster_df = airbnb_df[
        ['id', 'name', 'property_type', 'room_type', 'latitude', 'longitude', 'accommodates', 'amenities',
         'price']]

    cluster_df = cluster_df.dropna(subset=['price'])

    clustering_input_df = {
        'id': id,
        'name': listing_title,
        'property_type': property_type,
        'room_type': room_type,
        'latitude': lat,
        'longitude': lon,
        'accommodates': accommodates,
        'amenities': [amenities],
        'price': actual_predicted_price
    }

    # Merge clustering_input_df into cluster_df using only the necessary features above, and without preprocessing

    cluster_df = pd.concat([cluster_df, pd.DataFrame(clustering_input_df, index=[0])], ignore_index=True)

    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from geopy.distance import great_circle

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

    kmeans = KMeans(n_clusters=15, random_state=42)  # Adjust n_clusters as needed
    cluster_df['cluster'] = kmeans.fit_predict(features_scaled)

    # Example usage: find similar listings for a randomly selected target listing
    target_listing_id = id

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
    same_cluster_listings['within_radius'] = same_cluster_listings.apply(is_within_radius,
                                                                         target_location=target_location,
                                                                         radius_km=2, axis=1)

    # Filter listings that are within the 2km radius
    within_radius_listings = same_cluster_listings[same_cluster_listings['within_radius']]

    # Drop the 'within_radius' column as it's no longer needed
    within_radius_listings = within_radius_listings.drop(columns=['within_radius'])

    # Exclude the target listing from the final result
    final_result = within_radius_listings[within_radius_listings['id'] != target_listing_id]

    print(f'{final_result.shape[0]} similar listings found nearby.')
    st.write(f"{final_result.shape[0]} similar listings found nearby.")

    # Decode the label encoding for 'property_type' and 'room_type'
    final_result['property_type'] = label_encoder_property_type.inverse_transform(
        final_result['property_type_encoded'])
    final_result['room_type'] = label_encoder_room_type.inverse_transform(
        final_result['room_type_encoded'])

    # Display specified columns
    display_columns = ['name', 'property_type', 'room_type', 'accommodates', 'amenities', 'latitude',
                       'longitude',
                       'price']

    result = final_result[display_columns]

    # Display the result
    st.write(result.head(5))




