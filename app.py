import streamlit as st
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
from groq import Groq




model=pickle.load(open("model.pkl","rb"))
state=pickle.load(open("state.pkl","rb"))
crop=pickle.load(open("crop.pkl","rb"))
season=pickle.load(open("season.pkl","rb"))
soil=pickle.load(open("soil.pkl","rb"))


# Initialize session state for storing input values
if "farm_data" not in st.session_state:
    st.session_state.farm_data = {}

api_key = os.getenv("GROQ_API_KEY", "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")

client = Groq(api_key=api_key)

def get_chat_response(user_input, farm_data=None, language="en"):
    """
    language: 'en' for English, 'ta' for Tamil
    farm_data: dict containing crop, state, soil, season, area, rainfall, temperature, etc.
    """
    
    # Build context from farm data
    context = ""
    if farm_data:
        context = f"""
        Farmer's Current Farm Details:
        - Crop: {farm_data.get('crop', 'Not selected')}
        - State: {farm_data.get('state', 'Not selected')}
        - Soil Type: {farm_data.get('soil', 'Not selected')}
        - Season: {farm_data.get('season', 'Not selected')}
        - Farm Area: {farm_data.get('area', 'N/A')} hectares
        - Rainfall: {farm_data.get('rainfall', 'N/A')} mm
        - Temperature: {farm_data.get('temperature', 'N/A')} °C
        - Humidity: {farm_data.get('humidity', 'N/A')} %
        - Nitrogen (N): {farm_data.get('nitrogen', 'N/A')}
        - Phosphorus (P): {farm_data.get('phosphorus', 'N/A')}
        - Potassium (K): {farm_data.get('potassium', 'N/A')}
        - Predicted Yield: {farm_data.get('yield_pred', 'N/A')} quintal/hectare
        - Estimated Profit: ₹{farm_data.get('profit', 'N/A')}
        """
    
    system_prompt = f"""
    You are an AI assistant for Indian farmers. 
    Based on the farmer's specific details provided below, give personalized recommendations for:
    - Profit maximization
    - Fertilizer recommendations
    - Weather intelligence
    - Crop-specific advice
    - Soil management tips
    
    {context}
    
    Respond in Tamil if user asks in Tamil, otherwise in English.
    Keep answers practical, clear, and specific to their farm conditions.
    Provide actionable advice they can implement immediately.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Groq fast model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"



st.sidebar.title("Crop Yield Prediction App")
st.sidebar.markdown("This app predicts the crop yield based on various factors.")

page=st.sidebar.radio("Select Page",["Home","Predict","AI CHATBOT"])

if page == "Home":
    st.title("🌽Crop Yield Prediction")
    st.markdown("Welcome to the Crop Yield Prediction App!")
    st.markdown("Use the sidebar to navigate between pages.")

elif page == "Predict":
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
       
       price_per_quintal = st.number_input("Selling Price (₹ per quintal)", min_value=0.0)
       cost_per_hectare = st.number_input("Cost of Cultivation (₹ per hectare)", min_value=0.0)

       scenario = st.selectbox("Select Climate Scenario", 
                        ["Normal", "Drought", "Flood", "Heatwave", "Cold Spell"])

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
              revenue = production * price_per_quintal
              cost = cost_per_hectare * area
              profit = revenue - cost

              # Store farm data in session state
              st.session_state.farm_data = {
                  "crop": crop_selected,
                  "state": state_selected,
                  "soil": soil_selected,
                  "season": season_selected,
                  "area": area,
                  "rainfall": rainfall,
                  "temperature": temperature,
                  "humidity": humidity,
                  "nitrogen": nitrogen,
                  "phosphorus": phosphorus,
                  "potassium": potassium,
                  "yield_pred": round(yield_pred, 2),
                  "production": round(production, 2),
                  "profit": round(profit, 2),
                  "price_per_quintal": price_per_quintal,
                  "scenario": scenario
              }

              st.success(f"🌾 Predicted Yield: {yield_pred:.2f} quintal/hectare")
              st.info(f"📦 Estimated Production: {production:.2f} quintal")
              st.warning(f"💰 Estimated Profit: ₹{profit:.2f}")

              st.subheader("🌦 Climate Simulator")
              
              
              if scenario == "Drought":
                     rainfall = 50
                     humidity = 30
              elif scenario == "Flood":
                     rainfall = 400
                     humidity = 90
              elif scenario == "Heatwave":
                     temperature = 40
                     humidity = 20
              elif scenario == "Cold Spell":
                     temperature = 10
                     humidity = 80
              else:
                     rainfall = 200
                     temperature = 25
                     humidity = 60

              st.write(f"🌧 Rainfall: {rainfall} mm | 🌡 Temperature: {temperature} °C | 💧 Humidity: {humidity}%")
              
              
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

              
              
              
              
elif page == "AI CHATBOT":
       st.title("🌾 Crop Yield Predictor + AI Chatbot")

       # Chatbot section
       st.subheader("🤖 AI Farming Assistant (Tamil + English)")
       
       # Display stored farm data if available
       if st.session_state.farm_data:
           st.info("📊 **Your Current Farm Settings:**")
           col1, col2, col3 = st.columns(3)
           
           with col1:
               st.write(f"🌾 **Crop:** {st.session_state.farm_data.get('crop', 'N/A')}")
               st.write(f"🏞️ **State:** {st.session_state.farm_data.get('state', 'N/A')}")
               st.write(f"🌱 **Soil:** {st.session_state.farm_data.get('soil', 'N/A')}")
               
           with col2:
               st.write(f"📅 **Season:** {st.session_state.farm_data.get('season', 'N/A')}")
               st.write(f"📍 **Area:** {st.session_state.farm_data.get('area', 'N/A')} ha")
               st.write(f"🌧️ **Rainfall:** {st.session_state.farm_data.get('rainfall', 'N/A')} mm")
               
           with col3:
               st.write(f"🌾 **Yield:** {st.session_state.farm_data.get('yield_pred', 'N/A')} qt/ha")
               st.write(f"💰 **Profit:** ₹{st.session_state.farm_data.get('profit', 'N/A')}")
               st.write(f"📦 **Production:** {st.session_state.farm_data.get('production', 'N/A')} qt")
       else:
           st.warning("⚠️ No farm data available. Please go to 'Predict' page and click 'Predict Yield' first!")

       st.subheader("💬 Ask Your Questions")
       user_input = st.text_input("Ask about profit, fertilizer, or weather:")

       if user_input:
              # Detect Tamil vs English (simple heuristic)
              if any("\u0B80" <= ch <= "\u0BFF" for ch in user_input):
                     lang = "ta"
              else:
                     lang = "en"

              # Pass farm data to chatbot
              response = get_chat_response(user_input, farm_data=st.session_state.farm_data, language=lang)
              st.write("### 🤖 Chatbot Response:")
              st.success(response)
       else:
              st.info("Enter a question above to get a chatbot response.")
