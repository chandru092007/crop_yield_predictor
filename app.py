import streamlit as st
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np





model=pickle.load(open("model.pkl","rb"))
state=pickle.load(open("state.pkl","rb"))
crop=pickle.load(open("crop.pkl","rb"))
season=pickle.load(open("season.pkl","rb"))
soil=pickle.load(open("soil.pkl","rb"))

st.title("🌽Crop Yield Prediction")

state_selected = st.selectbox("Select State", list(state.classes_))
soil_selected = st.selectbox("Select Soil Type", list(soil.classes_))
crop_selected = st.selectbox("Select Crop", list(crop.classes_))
season_selected = st.selectbox("Select Season", list(season.classes_))

area = st.number_input("Area (hectare)", min_value=1.0)

rainfall = st.slider("Rainfall (mm)", 0, 500, 200)
temperature = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 60)

nitrogen = st.slider("Nitrogen (N)", 0, 150, 90)
phosphorus = st.slider("Phosphorus (P)", 0, 100, 40)
potassium = st.slider("Potassium (K)", 0, 100, 40)

state_enc = state.transform([state_selected])[0]
crop_enc = crop.transform([crop_selected])[0]
season_enc = season.transform([season_selected])[0]
soil_enc = soil.transform([soil_selected])[0]


if st.button("Predict Yield"):
    
    sample = np.array([[state_enc,crop_enc,season_enc,soil_enc,
                        area,rainfall,temperature,
                        humidity,nitrogen,phosphorus,potassium]])

    prediction = model.predict(sample)

    yield_pred = prediction[0]
    production = yield_pred * area

    st.success(f"🌾 Predicted Yield: {yield_pred:.2f} quintal/hectare")
    st.info(f"📦 Estimated Production: {production:.2f} quintal")

    # Yield visualization
    st.subheader("📊 Yield Distribution")

    fig, ax = plt.subplots()

    labels = ["Yield per hectare", "Total Production"]
    values = [yield_pred, production]

    colors = ["#22c55e", "#f97316"]
    explode = (0.1, 0)

    ax.pie(values,
           labels=labels,
           autopct='%1.1f%%',
           colors=colors,
           explode=explode,
           startangle=90)

    ax.legend(labels, loc="upper right")
    ax.set_title("Yield vs Production")

    st.pyplot(fig)