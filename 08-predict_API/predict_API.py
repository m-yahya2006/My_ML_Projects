
from sklearn.datasets import load_breast_cancer
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI

app = FastAPI()


data = load_breast_cancer()
X=data.data
y=data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=42)
model.fit(X_scaled,y)

class Features(BaseModel):
    features : list[float]

@app.post("/predict")    

def predict(data : Features):
    if len(data.features)!=30:
        return {"error": "Exactly 30 features are required."}
    features = np.array(data.features).reshape(1 , -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    label = "benign" if prediction ==1 else "malignant"
    return {"prediction" : label}        

