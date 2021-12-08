"""
Explanation of the dataset

Variables: ORDER: Sequence of clicks during one session

COUNTRY: 1-Australia 2-Austria 3-Belgium 4-British Virgin Islands 5-Cayman Islands 6-Christmas Island 7-Croatia 8-Cyprus 9-Czech Republic 10-Denmark 11-Estonia 12-unidentified 13-Faroe Islands 14-Finland 15-France 16-Germany 17-Greece 18-Hungary 19-Iceland 20-India 21-Ireland 22-Italy 23-Latvia 24-Lithuania 25-Luxembourg 26-Mexico 27-Netherlands 28-Norway 29-Poland 30-Portugal 31-Romania 32-Russia 33-San Marino 34-Slovakia 35-Slovenia 36-Spain 37-Sweden 38-Switzerland 39-Ukraine 40-United Arab Emirates 41-United Kingdom 42-USA 43-biz (.biz) 44-com (.com) 45-int (.int) 46-net (.net) 47-org (*.org)

SESSION ID: variable indicating session id (short record)

PAGE 1 (MAIN CATEGORY) -> concerns the main product category: 1-trousers 2-skirts 3-blouses 4-sale

PAGE 2 (CLOTHING MODEL) -> contains information about the code for each product (217 products)

COLOUR -> colour of product 1-beige 2-black 3-blue 4-brown 5-burgundy 6-gray 7-green 8-navy blue 9-of many colors 10-olive 11-pink 12-red 13-violet 14-white

LOCATION -> photo location on the page, the screen has been divided into six parts: 1-top left 2-top in the middle 3-top right 4-bottom left 5-bottom in the middle 6-bottom right

MODEL PHOTOGRAPHY -> variable with two categories: 1-en face 2-profile

PRICE -> price in US dollars

PRICE 2 -> variable informing whether the price of a particular product is higher than the average price for the entire product category 1-yes 2-no
"""

import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import sklearn

# Config and setup
st.set_page_config(layout="wide", page_title="Blue Jeans Dashboard")

st.header("Blue Jeans Dashboard")

# Reading the dataset
clicks = pd.read_csv('clicks.csv', sep=";")

st.header("Data Exploration")

image1 = Image.open('graph3.PNG')
image2 = Image.open('graph4.PNG')
st.write("#### Amount of clothing type sold on a specific day")
st.image(image1)
st.write("#### Average price of items grouped by colours")
st.image(image2)

st.header("Clicks Prediction")
st.write("#### Configure filters and press the button to get the predicted order of clicks")

st.write("##### Explanation of the dataset\n"
         "* PAGE 1 (MAIN CATEGORY) -> concerns the main product category: 1-trousers "
         "2-skirts 3-blouses 4-sale \n"

         "* COLOUR -> colour of product 1-beige 2-black 3-blue 4-brown 5-burgundy 6-gray "
         "7-green 8-navy blue 9-of many colors 10-olive 11-pink 12-red 13-violet 14-white \n"
         "* LOCATION -> photo location "
         "on the page, the screen has been divided into six parts: 1-top left 2-top in the middle 3-top right 4-bottom "
         "left 5-bottom in the middle 6-bottom right \n"
         "* MODEL PHOTOGRAPHY -> variable with two categories: 1-en face 2-profile ")

# Setting up filters
price = st.slider('Price (US dollars)', 1, 100, 50)
colour = st.slider('Colour id', 1, 14, 1)
location = st.slider('Location id', 1, 6, 1)
page1 = st.slider('Product category', 1, 4, 1)
model_photography = st.selectbox('Photo type id', (1, 2))


# colour = st.selectbox('Colour id', ('beige', 'black', 'blue', 'brown', 'burgundy','gray','green','navy blue' ,'of many colors','olive', 'pink', 'red','violet', 'white') , index=1)
# location = st.selectbox('Location of an item ', ('top left', 'top in the middle', 'top right','bottom left', 'bottom in the middle', 'bottom right'))
# model_photography = st.selectbox('Photo type',('en face ','profile'))
# page1 = st.selectbox('Product category', ('trousers ','skirts ','blouses ','sale'))


# Accepting the user input
def user_input_features(price, colour, location, model_photography, page1):
    data = {
        'price': price, 'colour': colour, 'location': location,'page1': page1, 'model photography': model_photography,
        }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features(price, colour, location, model_photography, page1)


# Combines user input features with entire routes dataset
clicks = clicks.drop(
    columns=['order', 'year', 'month', 'day', 'country', 'session ID', 'page 2 (clothing model)', 'price 2', 'page','page 1 (main category)'])

df = pd.concat([input_df, clicks], axis=0)

df = df[:1]  # Selects only the first row (the user input data)"


if st.button('Predict'):
    # form.empty()
    with st.spinner('Processing...'):
        # Reads in saved classification model
        load_clf = pickle.load(open('clicks_clf.pkl', 'rb'))

        # Apply model to make predictions
        prediction = load_clf.predict(df)

        st.write("#### Order of clicks will be the next")
        st.write(prediction)
