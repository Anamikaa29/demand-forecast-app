
import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

st.title("Walmart-Style Product Demand Forecaster")

st.write("Upload a CSV file with columns `ds`, `y`, and `product`. Select a product to forecast future demand.")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if {'ds', 'y', 'product'}.issubset(df.columns):
        df['ds'] = pd.to_datetime(df['ds'])
        product_list = df['product'].unique()
        selected_product = st.selectbox("Select a product", product_list)
        df_product = df[df['product'] == selected_product]

        model = Prophet()
        model.fit(df_product[['ds', 'y']])

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.subheader(f"Forecast for {selected_product}")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("Forecast Table (Next 30 Days)")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
    else:
        st.error("CSV must have columns: ds, y, and product.")
