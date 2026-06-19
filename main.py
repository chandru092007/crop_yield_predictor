import pandas as pd 
import numpy as np 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
import pickle



df=pd.read_csv('india_crop_yield_20000_rows.csv')
state=LabelEncoder()
soil=LabelEncoder()
crop=LabelEncoder()
season=LabelEncoder()

df["State"]=state.fit_transform(df["State"])
df["Soil_Type"]=soil.fit_transform(df["Soil_Type"])
df["Crop"]=crop.fit_transform(df["Crop"])
df["Season"]=season.fit_transform(df["Season"])


df["yield"] = df["Production"] / df["Area"]

x = df.drop(columns=["yield", "Production"])
y = df["yield"]


model = StackingRegressor(
    estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=200, random_state=42))],
    final_estimator=GradientBoostingRegressor(n_estimators=200, random_state=42),
    cv=5
)
model.fit(x, y)

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(state,open('state.pkl','wb'))
pickle.dump(soil,open('soil.pkl','wb')) 
pickle.dump(crop,open('crop.pkl','wb'))
pickle.dump(season,open('season.pkl','wb'))
