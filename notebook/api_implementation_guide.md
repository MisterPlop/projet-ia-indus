# 🚀 Guide d'Implémentation API - Modèle Airbnb

## 1. Script predict.py

Créez un fichier `predict.py` :

```python
# predict.py
import pickle
import pandas as pd
import argparse

def load_model(model_path='airbnb_simple_model.pkl'):
    """Charger le modèle sauvegardé"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_price(model_data, listing_data):
    """Prédire le prix d'un logement"""
    # Extraire les composants
    model = model_data['model']
    encoders = model_data['encoders']
    features = model_data['features']
    
    # Convertir en DataFrame
    df_input = pd.DataFrame([listing_data])
    
    # Encoder les variables catégorielles
    for col in ['neighbourhood_group', 'room_type']:
        if col in df_input.columns:
            df_input[col] = encoders[col].transform(df_input[col])
    
    # Prédire
    price = model.predict(df_input[features])[0]
    return round(price, 2)

if __name__ == "__main__":
    # Parser pour ligne de commande
    parser = argparse.ArgumentParser(description='Prédire le prix Airbnb')
    parser.add_argument('--neighbourhood_group', required=True)
    parser.add_argument('--room_type', required=True)
    parser.add_argument('--latitude', type=float, required=True)
    parser.add_argument('--longitude', type=float, required=True)
    parser.add_argument('--minimum_nights', type=int, default=1)
    parser.add_argument('--number_of_reviews', type=int, default=0)
    parser.add_argument('--availability_365', type=int, default=365)
    
    args = parser.parse_args()
    
    # Charger le modèle
    model_data = load_model()
    
    # Faire la prédiction
    listing = {
        'neighbourhood_group': args.neighbourhood_group,
        'room_type': args.room_type,
        'latitude': args.latitude,
        'longitude': args.longitude,
        'minimum_nights': args.minimum_nights,
        'number_of_reviews': args.number_of_reviews,
        'availability_365': args.availability_365
    }
    
    predicted_price = predict_price(model_data, listing)
    print(f"Prix prédit : ${predicted_price}")
```

### Utilisation du script :

```bash
python predict.py \
    --neighbourhood_group "Manhattan" \
    --room_type "Entire home/apt" \
    --latitude 40.7589 \
    --longitude -73.9851 \
    --minimum_nights 2 \
    --number_of_reviews 10 \
    --availability_365 365
```

## 2. API FastAPI

Créez un fichier `app.py` :

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import Optional

# Initialiser FastAPI
app = FastAPI(title="Airbnb Price Predictor", version="1.0.0")

# Modèle de données pour la requête
class ListingData(BaseModel):
    neighbourhood_group: str
    room_type: str
    latitude: float
    longitude: float
    minimum_nights: int = 1
    number_of_reviews: int = 0
    availability_365: int = 365

# Modèle de données pour la réponse
class PriceResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"

# Charger le modèle au démarrage
MODEL_DATA = None

@app.on_event("startup")
async def load_model():
    global MODEL_DATA
    try:
        with open('airbnb_simple_model.pkl', 'rb') as f:
            MODEL_DATA = pickle.load(f)
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")

# Endpoint principal de prédiction
@app.post("/predict", response_model=PriceResponse)
async def predict_price(listing: ListingData):
    if MODEL_DATA is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        # Extraire les composants du modèle
        model = MODEL_DATA['model']
        encoders = MODEL_DATA['encoders']
        features = MODEL_DATA['features']
        
        # Convertir en dictionnaire puis DataFrame
        listing_dict = listing.dict()
        df_input = pd.DataFrame([listing_dict])
        
        # Encoder les variables catégorielles
        for col in ['neighbourhood_group', 'room_type']:
            if col in df_input.columns:
                df_input[col] = encoders[col].transform(df_input[col])
        
        # Prédire
        price = model.predict(df_input[features])[0]
        
        return PriceResponse(predicted_price=round(price, 2))
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Valeur invalide : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# Endpoint d'info sur le modèle
@app.get("/model/info")
async def model_info():
    if MODEL_DATA is None:
        return {"status": "error", "message": "Modèle non chargé"}
    
    return {
        "status": "loaded",
        "features": MODEL_DATA['features'],
        "neighbourhood_groups": list(MODEL_DATA['encoders']['neighbourhood_group'].classes_),
        "room_types": list(MODEL_DATA['encoders']['room_type'].classes_)
    }

# Endpoint de santé
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "API fonctionnelle"}

# Endpoint racine avec documentation
@app.get("/")
async def root():
    return {
        "message": "API de Prédiction des Prix Airbnb NYC",
        "endpoints": {
            "predict": "/predict - POST - Prédire le prix d'un logement",
            "model_info": "/model/info - GET - Information sur le modèle",
            "health": "/health - GET - Statut de l'API",
            "docs": "/docs - GET - Documentation Swagger"
        }
    }
```

## 3. Fichier requirements.txt

```text
fastapi==0.104.1
uvicorn==0.24.0
pandas==1.5.3
scikit-learn==1.3.0
numpy==1.24.3
pydantic==2.4.2
```

## 4. Démarrage de l'API

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

L'API sera disponible sur : http://localhost:8000

## 5. Utilisation de l'API

### Avec curl :

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "neighbourhood_group": "Manhattan",
    "room_type": "Entire home/apt",
    "latitude": 40.7589,
    "longitude": -73.9851,
    "minimum_nights": 2,
    "number_of_reviews": 10,
    "availability_365": 365
}'
```

### Avec Python :

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "neighbourhood_group": "Manhattan",
    "room_type": "Entire home/apt",
    "latitude": 40.7589,
    "longitude": -73.9851,
    "minimum_nights": 2,
    "number_of_reviews": 10,
    "availability_365": 365
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prix prédit : ${result['predicted_price']}")
```

## 6. Docker (Optionnel)

Créez un `Dockerfile` :

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY airbnb_simple_model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Construire l'image
docker build -t airbnb-api .

# Lancer le conteneur
docker run -p 8000:8000 airbnb-api
```

## 7. Tests de l'API

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "neighbourhood_group": "Manhattan",
        "room_type": "Entire home/apt",
        "latitude": 40.7589,
        "longitude": -73.9851,
        "minimum_nights": 2,
        "number_of_reviews": 10,
        "availability_365": 365
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"
```

## 8. Documentation

Une fois l'API lancée, la documentation Swagger est disponible automatiquement sur :
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## 9. Sécurité (Production)

Pour la production, ajoutez :

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Vérifier le token ici
    pass

@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict_price(listing: ListingData):
    # ... code de prédiction
```

## 10. Monitoring

Ajoutez des métriques :

```python
import time
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    REQUEST_COUNT.inc()
    response = await call_next(request)
    process_time = time.time() - start_time
    REQUEST_LATENCY.observe(process_time)
    return response
```

Tu veux que je détaille une partie spécifique de l'implémentation API ?
