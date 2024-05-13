import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# Title and description
st.title("Laptop Price Predictor")
st.write("Welcome to the Laptop Price Predictor app! Select the specifications of your desired laptop and we'll predict its price.")

# Sidebar with input options
st.sidebar.title("Configure Your Laptop")
company = st.sidebar.selectbox('Brand', df['Company'].unique())
type = st.sidebar.selectbox('Type', df['TypeName'].unique())
ram = st.sidebar.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.sidebar.number_input('Weight of the Laptop')
touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.sidebar.selectbox('IPS', ['No', 'Yes'])
screen_size = st.sidebar.number_input('Screen Size')
resolution = st.sidebar.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.sidebar.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.sidebar.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.sidebar.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.sidebar.selectbox('GPU', df['Gpu brand'].unique())
os = st.sidebar.selectbox('OS', df['os'].unique())

# Space for laptop image
st.image("Designer.png", use_column_width=True)

# Prediction button
if st.sidebar.button('Predict Price'):
    # Prepare input data for prediction
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, 12)

    # Make prediction and display result
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.title("Predicted Price:")
    st.write(f"The predicted price of this configuration is {predicted_price}")
