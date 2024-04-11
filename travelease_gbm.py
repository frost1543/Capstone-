import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import openai
from openai import OpenAI

# Setting OpenAI API key
api_key = "1sk-bkjP33GOgjSXEwQilG8BT3BlbkFJ41ShZ7evhd0T8jz2M6FP"
client = OpenAI(api_key=api_key)

# Loading the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_listings.csv", encoding='latin1') 
    return data

# Getting neighbourhoods for the selected city
def get_neighbourhoods(data, city):
    neighbourhoods = data[data['city'] == city]['neighbourhood'].unique()
    return neighbourhoods

# Training the model
@st.cache_data
def train_model(data):
    # Removing rows with missing values
    data.dropna(inplace=True)

    # Selecting relevant features and target variable
    X = data[['city', 'neighbourhood', 'property_type', 'accommodates', 'bedrooms']]
    y = data['price']

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    feature_names = encoder.get_feature_names_out(input_features=X.columns)
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=feature_names)  

    # Spliting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)

    # hyperparameter grid 
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees
        'max_depth': [3, 5, 7],  # Maximum depth of each tree
        'learning_rate': [0.1, 0.01, 0.001],  # Learning rate
        'gamma': [0, 0.1, 1],  # Regularization parameter
        'lambda': [0, 0.1, 1],  
        'alpha': [0, 0.1, 1]  
    }

    
    model = XGBRegressor(random_state=42)

    # Perform hyperparameter tuning using GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    
    best_model = grid_search.best_estimator_

    return best_model, encoder

# Generating travel tips using OpenAI API
def generate_travel_tips(city, nationality, start_date, end_date):
    prompt = f"As a {nationality} traveler planning a trip to {city} from {start_date} to {end_date}, I need to know the visa requirements, travel insurance, packing essentials, and safety tips for my travel dates."
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Calculating total price based on date range and predicted price
def calculate_total_price(start_date, end_date, predicted_price):
    delta = end_date - start_date
    num_days = delta.days
    total_price = num_days * predicted_price
    return total_price

# Main function to run the Streamlit app
def main():
    st.title("Accommodation Price Prediction and Travel Tips")

    
    data = load_data()

    # Training the model
    model, encoder = train_model(data)

    # Sidebar inputs for user
    st.sidebar.header('User Input')
    city = st.sidebar.selectbox('City', data['city'].unique())
    nationality = st.sidebar.selectbox('Nationality', ['USA', 'UK', 'Canada', 'Australia', 'Germany'])
    start_date = st.sidebar.date_input('Start Date')
    end_date = st.sidebar.date_input('End Date')

    # Validate end date is greater than start date
    if end_date <= start_date:
        st.sidebar.error("End date must be greater than start date.")
    else:
        neighbourhoods = get_neighbourhoods(data, city)
        neighbourhood = st.sidebar.selectbox('Neighbourhood', neighbourhoods)
        property_type = st.sidebar.selectbox('Property Type', data['property_type'].unique())
        accommodates = st.sidebar.slider('Accommodates', min_value=1, max_value=20, value=4)
        bedrooms = st.sidebar.slider('Bedrooms', min_value=0, max_value=10, value=1)

        # Check if user has submitted the form
        if st.sidebar.button('Submit'):
            # One-hot encode user input
            input_data = pd.DataFrame({
                'city': [city],
                'neighbourhood': [neighbourhood],
                'property_type': [property_type],
                'accommodates': [accommodates],
                'bedrooms': [bedrooms]
            })
            input_encoded = encoder.transform(input_data)

            # Get feature names after one-hot encoding
            feature_names_encoded = encoder.get_feature_names_out(input_data.columns)

            
            input_encoded_df = pd.DataFrame(input_encoded.toarray(), columns=feature_names_encoded)  

            # Make prediction
            prediction = model.predict(input_encoded_df)

            # Display price prediction
            st.subheader('Price Prediction')
            st.write(f'The predicted price for the accommodation is: ${prediction[0]:.2f}')

            # Calculate total price based on date range and predicted price
            total_price = calculate_total_price(start_date, end_date, prediction[0])

            # Display total price
            st.subheader('Total Price')
            st.write(f'The total price for your stay is: ${total_price:.2f}')

            # Generate and display travel tips
            st.subheader('Travel Tips')
            travel_tips = generate_travel_tips(city, nationality, start_date, end_date)
            st.write(travel_tips)


if __name__ == "__main__":
    main()
