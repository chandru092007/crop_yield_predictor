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



api_key = os.getenv("GROQ_API_KEY", "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")

client = Groq(api_key=api_key)

def get_chat_response(user_input, language="en"):
    """
    language: 'en' for English, 'ta' for Tamil
    """
    system_prompt = """
    You are an AI assistant for farmers. 
    Provide profit prediction, fertilizer recommendation, and weather intelligence.
    Respond in Tamil if user asks in Tamil, otherwise in English.
    Keep answers practical and clear.
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
              
              
              
elif page == "AI CHATBOT":
       st.title("🌾 Crop Yield Predictor + AI Chatbot")

# Chatbot section
       st.subheader("🤖 AI Farming Assistant (Tamil + English)")

       user_input = st.text_input("Ask about profit, fertilizer, or weather:")

       if user_input:
              # Detect Tamil vs English (simple heuristic)
              if any("\u0B80" <= ch <= "\u0BFF" for ch in user_input):
                     lang = "ta"
              else:
                     lang = "en"

              response = get_chat_response(user_input, language=lang)
              st.write("### Chatbot Response:")
              st.success(response)
       else:
              st.info("Enter a question above to get a chatbot response.")
