import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import os


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Actual Value - A Car Price Predictor",
    layout="wide"
)


# -------------------- LOAD MODEL --------------------
model_path = os.path.join(os.getcwd(), "model.pkl")
model = pk.load(open("model.pkl", "rb"))
cars_data = pd.read_csv('Cardetails.csv')


# -------------------- CUSTOM STYLING --------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
    font-family: 'Segoe UI', sans-serif;
}

/* Title Styling */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}

/* Advertisement Boxes */
.ad-banner {
    background-color: white;
    border: 1px dashed #cbd5e1;
    padding: 15px;
    text-align: center;
    border-radius: 8px;
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 25px;
}

.ad-sidebar {
    background-color: white;
    border: 1px dashed #cbd5e1;
    padding: 25px;
    text-align: center;
    border-radius: 10px;
    color: #6b7280;
    font-size: 14px;
}

/* Section Header */
h3 {
    color: #111827;
    font-weight: 600;
}

/* Buttons - Reduced Size */
.stButton > button {
    background: linear-gradient(to right, #2563eb, #3b82f6);
    color: white;
    font-size: 15px;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 8px;
    border: none;
    width: 220px;
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    background: linear-gradient(to right, #1e40af, #2563eb);
    transform: translateY(-2px);
}

/* Result Box */
.result-box {
    background: linear-gradient(135deg, #10b981, #059669);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    color: white;
    font-weight: bold;
    margin-top: 30px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 60px;
    color: #6b7280;
    font-size: 14px;
}

/* Padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

</style>
""", unsafe_allow_html=True)


# -------------------- TITLE SECTION --------------------
st.markdown("<div class='title'>Predecting current value of your Car</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Know Your Car’s True Market Value — Fast, Fair & Transparent</div>", unsafe_allow_html=True)




# -------------------- DATA PREPROCESS --------------------
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)


# -------------------- MAIN LAYOUT (CONTENT + SIDEBAR AD) --------------------
main_col, ad_col = st.columns([3,1])

with main_col:

    st.subheader("Enter Car Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.selectbox('Car Brand', cars_data['name'].unique())
        year = st.slider('Manufactured Year', 1994, 2024)
        fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
        mileage = st.slider('Mileage (km/l)', 10, 40)

    with col2:
        km_driven = st.slider('Kilometers Driven', 11, 200000)
        seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
        transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
        engine = st.slider('Engine CC', 700, 5000)

    with col3:
        owner = st.selectbox('Owner Type', cars_data['owner'].unique())
        max_power = st.slider('Max Power (bhp)', 0, 200)
        seats = st.slider('Number of Seats', 5, 10)



# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Car Price"):

    with st.spinner("Calculating best market price... ⏳"):

        input_data_model = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission,
              owner, mileage, engine, max_power, seats]],
            columns=['name','year','km_driven','fuel','seller_type',
                     'transmission','owner','mileage','engine',
                     'max_power','seats']
        )

        # Encoding categorical values
        input_data_model['owner'].replace(
            ['First Owner', 'Second Owner', 'Third Owner',
             'Fourth & Above Owner', 'Test Drive Car'],
            [1,2,3,4,5], inplace=True)

        input_data_model['fuel'].replace(
            ['Diesel', 'Petrol', 'LPG', 'CNG'],
            [1,2,3,4], inplace=True)

        input_data_model['seller_type'].replace(
            ['Individual', 'Dealer', 'Trustmark Dealer'],
            [1,2,3], inplace=True)

        input_data_model['transmission'].replace(
            ['Manual', 'Automatic'],
            [1,2], inplace=True)

        input_data_model['name'].replace(
            ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
             'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
             'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
             'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
             'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
            list(range(1,32)), inplace=True)

        car_price = model.predict(input_data_model)[0]


    # Display Result
    st.markdown(f"""
        <div class='result-box'>
         Estimated Car Price: ₹ {car_price:,.2f}
        </div>
    """, unsafe_allow_html=True)


# -------------------- FOOTER --------------------
st.markdown("""
<div class='footer'>
Developed by Ashutosh Kumar Gautam | Machine Learning Portfolio Project | 2026
</div>

""", unsafe_allow_html=True)

