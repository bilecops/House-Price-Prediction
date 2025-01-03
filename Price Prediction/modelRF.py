import streamlit as st
from joblib import load
import numpy as np
import math
from datetime import datetime

# Load the trained Random Forest model for Landed
model_path = "RandomForest_Landed.joblib"  # Adjust the path as needed
model = load(model_path)

# Title of the app for Landed
st.title("Random Forest Prediction for Landed Prices")

# Class descriptions for each feature for Landed property
st.markdown(""" 
**Feature Descriptions:**
1. **Log Land/Parcel Area:** Logarithmic transformation of the land or parcel area. 
   (Log of the original square footage.)
2. **Log Main Floor Area:** Logarithmic transformation of the main floor area.
   (Log of the original square footage.)
3. **Transaction Date (Ordinal):** The date of the property transaction converted to an ordinal date (integer form).
4. **Property Type:** Categorical variable indicating the type of property (e.g., 'Cluster House', 'Detached', 'Low-Cost House').
5. **Mukim:** A geographical region code representing the location of the property (e.g., 'Kuala Lumpur Town Centre', 'Mukim Ampang', etc.).
6. **Tenure:** Indicates the property tenure ('Freehold' or 'Leasehold').
""")

# Converter for the log-transformed features
def log_to_original(log_value):
    return math.exp(log_value)  # Convert log to original scale using exp

# Input features with class descriptions for Landed property
feature1 = st.number_input("Landed Log Land/Parcel Area", value=7.5, step=0.1)
original_feature1 = log_to_original(feature1)  # Show original value (square footage) for user information
st.write(f"Original Land/Parcel Area: {original_feature1:,.2f} square feet")

feature2 = st.number_input("Landed Log Main Floor Area", value=8.0, step=0.1)
original_feature2 = log_to_original(feature2)  # Show original value (square footage) for user information
st.write(f"Original Main Floor Area: {original_feature2:,.2f} square feet")

# Convert transaction date to ordinal for Landed property
feature3 = st.date_input("Landed Transaction Date", value=datetime(2020, 1, 1))
ordinal_feature3 = feature3.toordinal()
st.write(f"Transaction Date (Ordinal): {ordinal_feature3}")

# Property Type (Categorical feature with encoding for Landed property)
property_types = ['1 - 1 1/2 Storey Semi-Detached', '1 - 1 1/2 Storey Terraced',
 '2 - 2 1/2 Storey Semi-Detached', '2 - 2 1/2 Storey Terraced',
 'Cluster House', 'Detached', 'Low-Cost House']
property_type_mapping = {pt: idx for idx, pt in enumerate(property_types)}  # Create encoding map
feature4 = st.selectbox("Landed Property Type", options=property_types, index=0)
encoded_feature4 = property_type_mapping[feature4]  # Encode selected value

# Mukim (Geographical region, Categorical feature with encoding for Landed property)
mukims = ['Kuala Lumpur Town Centre', 'Mukim Ampang', 'Mukim Batu', 'Mukim Cheras',
          'Mukim Kuala Lumpur', 'Mukim Petaling', 'Mukim Setapak', 'Mukim Ulu Kelang']
mukim_mapping = {mukim: idx for idx, mukim in enumerate(mukims)}  # Create encoding map
feature5 = st.selectbox("Landed Mukim", options=mukims, index=0)
encoded_feature5 = mukim_mapping[feature5]  # Encode selected value

# Tenure (Categorical feature with encoding for Landed property)
tenures = ['Freehold', 'Leasehold']
tenure_mapping = {tenure: idx for idx, tenure in enumerate(tenures)}  # Create encoding map
feature6 = st.selectbox("Landed Tenure", options=tenures, index=0)
encoded_feature6 = tenure_mapping[feature6]  # Encode selected value

# Predict button for Landed property
if st.button("Predict Landed Price"):
    # Prepare input features as a NumPy array (including encoded categorical features)
    input_features = np.array([[encoded_feature4, encoded_feature5, encoded_feature6, feature1, feature2, ordinal_feature3]])

    # Perform prediction
    predicted_price = model.predict(input_features)[0]

    # Display the predicted price
    st.subheader(f"Landed Predicted Price (RM): {predicted_price:,.2f}")

# Load the trained Random Forest model for Highrise
model_patha = "RandomForest_Highrise.joblib"  # Adjust the path as needed
modela = load(model_patha)

# Title of the app for Highrise
st.title("Random Forest Prediction for Highrise Prices")

# Class descriptions for each feature for Highrise
st.markdown(""" 
**Feature Descriptions:**
1. **Log Land/Parcel Area:** Logarithmic transformation of the land or parcel area. 
   (Log of the original square footage.)
2. **Unit Level:** The level (or floor) of the unit in a highrise building.
3. **Transaction Date (Ordinal):** The date of the property transaction converted to an ordinal date (integer form).
4. **Property Type:** Categorical variable indicating the type of property (e.g., 'Condominium/Apartment').
5. **Mukim:** A geographical region code representing the location of the property (e.g., 'Kuala Lumpur Town Centre', 'Mukim Ampang', etc.).
6. **Tenure:** Indicates the property tenure ('Freehold' or 'Leasehold').
""")

# Input features with class descriptions for Highrise property
feature1a = st.number_input("Highrise Log Land/Parcel Area", value=7.5, step=0.1)
original_feature1a = log_to_original(feature1a)  # Show original value (square footage) for user information
st.write(f"Original Land/Parcel Area: {original_feature1a:,.2f} square feet")

feature2a = st.number_input("Highrise Unit Level", value=1.0, step=1.0)

# Convert transaction date to ordinal for Highrise property
feature3a = st.date_input("Highrise Transaction Date", value=datetime(2020, 1, 1))
ordinal_feature3a = feature3a.toordinal()
st.write(f"Transaction Date (Ordinal): {ordinal_feature3a}")

# Property Type (Categorical feature with encoding for Highrise property)
property_types_h = ['Condominium/Apartment', 'Flat', 'Low-Cost Flat', 'Town House']
property_type_mapping_h = {pt: idx for idx, pt in enumerate(property_types_h)}  # Create encoding map
feature4a = st.selectbox("Highrise Property Type", options=property_types_h, index=0)
encoded_feature4a = property_type_mapping_h[feature4a]  # Encode selected value

# Mukim (Geographical region, Categorical feature with encoding for Highrise property)
mukimsa = ['Kuala Lumpur Town Centre', 'Mukim Ampang', 'Mukim Batu', 'Mukim Cheras',
          'Mukim Kuala Lumpur', 'Mukim Petaling', 'Mukim Setapak', 'Mukim Ulu Kelang']
mukim_mappinga = {mukima: idx for idx, mukima in enumerate(mukimsa)}  # Create encoding map
feature5a = st.selectbox("Highrise Mukim", options=mukimsa, index=0)
encoded_feature5a = mukim_mappinga[feature5a]  # Encode selected value

# Tenure (Categorical feature with encoding for Highrise property)
tenuresa = ['Freehold', 'Leasehold']
tenure_mappinga = {tenurea: idx for idx, tenurea in enumerate(tenuresa)}  # Create encoding map
feature6a = st.selectbox("Highrise Tenure", options=tenuresa, index=0)
encoded_feature6a = tenure_mappinga[feature6a]  # Encode selected value

# Predict button for Highrise property
if st.button("Predict Highrise Price"):
    # Prepare input features as a NumPy array (including encoded categorical features)
    input_featuresa = np.array([[encoded_feature4a, encoded_feature5a, encoded_feature6a, feature2a, feature1a, ordinal_feature3a]])

    # Perform prediction
    predicted_price_h = modela.predict(input_featuresa)[0]

    # Display the predicted price
    st.subheader(f"Highrise Predicted Price (RM): {predicted_price_h:,.2f}")
