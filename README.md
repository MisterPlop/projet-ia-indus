# 🏠 Airbnb Price Predictor NYC

## 📖 Description

Ce projet implémente un système complet de prédiction des prix des logements Airbnb à New York City. Il utilise des techniques de machine learning (Gradient Boosting) pour fournir des prédictions rapides et relativement précises basées sur les caractéristiques du logement.

### 🎯 Objectifs

- Prédire le prix d'un logement Airbnb basé sur ses caractéristiques
- Offrir une interface web interactive avec Streamlit
- Créer une base solide pour un système de prédiction en production

## 📊 Dataset

Le modèle utilise le dataset "New York City Airbnb Open Data" :

- **Source** : [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- **Taille** : ~50k logements
- **Features** : Localisation, type de logement, prix, avis, etc.

## 📁 Structure du Projet

```
airbnb-price-predictor/
│
├── notebook/
│   ├── airbnb.ipynb              # Notebook d'entraînement
│   ├── README.md                 # Documentation du notebook
│   └── api_implementation_guide.md
│
├── pickles/
│   └── airbnb_simple_model.pkl   # Modèle sauvegardé
│
├── app.py                        # Interface Streamlit
├── predict.py                    # Script de prédiction
├── requirements.txt              # Dépendances
│
├── .github/
│   └── workflows/
│       └── test.yml              # CI/CD GitHub Actions
│
└── README.md                     # Ce fichier
```

## 🔧 Features du Modèle

Le modèle utilise 7 features principales :

1. **neighbourhood_group** : Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
2. **room_type** : Type de logement (Entire home/apt, Private room, Shared room)
3. **latitude** : Latitude
4. **longitude** : Longitude
5. **minimum_nights** : Nombre minimum de nuits
6. **number_of_reviews** : Nombre d'avis
7. **availability_365** : Disponibilité annuelle

## 📈 Performance

- **R² (Test)** : ~0.445
- **MAE** : ~$48.36
- **Temps d'entraînement** : < 30 secondes
- **Temps de prédiction** : < 1ms

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/MisterPlop/projet-ia-indus.git
cd projet-ia-indus
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Vérifier l'installation

```bash
python predict.py
```

## 💻 Utilisation

### 1. Script de prédiction en ligne de commande

```bash
python predict.py
```

Le script utilise des valeurs par défaut et affiche le prix prédit pour un exemple de logement.

### 2. Interface Streamlit

Lancez l'interface web interactive :

```bash
streamlit run app.py
```

L'interface sera disponible sur : http://localhost:8501

#### Fonctionnalités de l'interface :

- **Design moderne et responsive** avec couleurs Airbnb
- **Sélection interactive** des paramètres
- **Carte en temps réel** de la localisation
- **Calculs automatiques** (prix mensuel, revenus annuels)
- **Graphiques de comparaison**
- **Conseils d'optimisation** personnalisés

## 🔬 Tests

Exécutez les tests automatisés :

```bash
pytest
```

Les tests incluent :

- Validation du modèle
- Tests de l'API
- Tests de l'interface Streamlit

## 📚 Documentation

### 📓 Notebook

Le notebook `notebook/airbnb.ipynb` contient :

- Exploration des données
- Nettoyage et préprocessing
- Entraînement du modèle
- Évaluation des performances
- Visualisations

### 🎛️ Interface Streamlit

L'interface `app.py` offre :

- Saisie interactive des paramètres
- Visualisation en temps réel
- Export des résultats
- Mode debug pour développeurs

### 🔌 API REST

Consultez `notebook/api_implementation_guide.md` pour :

- Implémentation FastAPI complète
- Endpoints disponibles
- Authentification
- Déploiement Docker

## 🌐 Déploiement

### Option 1 : Streamlit Cloud

1. Push le code sur GitHub
2. Connectez votre repo à [Streamlit Cloud](https://streamlit.io/cloud)
3. Déploiement automatique !

### Option 2 : Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t airbnb-predictor .
docker run -p 8501:8501 airbnb-predictor
```

## 📄 Exemple d'utilisation

```python
# Exemple de prédiction
example = {
    'neighbourhood_group': 'Manhattan',
    'room_type': 'Entire home/apt',
    'latitude': 40.7589,
    'longitude': -73.9851,
    'minimum_nights': 2,
    'number_of_reviews': 10,
    'availability_365': 365
}

# Résultat attendu : ~$334
```

## 🤝 Contribution

1. Fork le repository
2. Créez une nouvelle branche (`git checkout -b feature/amélioration`)
3. Commit vos changes (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amélioration`)
5. Créez une Pull Request

## 📝 Roadmap

- [ ] Amélioration du modèle (XGBoost, Random Forest)
- [ ] Ajout de nouvelles features
- [ ] API avec authentification
- [ ] Cache Redis pour les prédictions
- [ ] Monitoring et métriques
- [ ] Tests automatisés plus complets
- [ ] Documentation API avec Swagger

## 📧 Contact

- **Auteurs** : VIVANT DYLAN, KONAK ALAN, ZOOGONES SYLVAIN, CHAOUKI BRAHIM

## 🙏 Remerciements

- [Kaggle](https://www.kaggle.com/) pour le dataset
- [Streamlit](https://streamlit.io/) pour l'interface web
- [FastAPI](https://fastapi.tiangolo.com/) pour l'API
- [scikit-learn](https://scikit-learn.org/) pour les modèles ML

---

_Made with ❤️ by Dylan, Alan, Sylvain & Brahim_
