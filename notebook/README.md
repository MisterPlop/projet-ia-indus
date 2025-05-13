# 🏠 Modèle de Prédiction des Prix Airbnb NYC

## 📖 Description

Ce projet implémente un modèle d'intelligence artificielle simple pour prédire les prix des logements Airbnb à New York City. Le modèle utilise un algorithme de Gradient Boosting pour fournir des prédictions rapides et relativement précises.

## 🎯 Objectifs

- Prédire le prix d'un logement Airbnb basé sur ses caractéristiques
- Fournir un modèle simple et rapide à implémenter
- Créer une base pour une API de prédiction

## 📊 Dataset

Le modèle utilise le dataset "New York City Airbnb Open Data" disponible sur Kaggle :
- **Source** : https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
- **Taille** : ~50k logements
- **Features** : Localisation, type de logement, prix, avis, etc.

## 📁 Structure du Projet

```
airbnb-price-predictor/
│
├── notebook/
│   └── airbnb_model_simple.ipynb   # Notebook d'entraînement
│
├── models/
│   └── airbnb_simple_model.pkl     # Modèle sauvegardé
│
├── api/
│   ├── app.py                      # Application FastAPI
│   ├── predict.py                  # Script de prédiction
│   └── requirements.txt            # Dépendances
│
├── tests/
│   └── test_model.py               # Tests unitaires
│
└── README.md
```

## 📚 Utilisation du Notebook

### 1. Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Exécution

1. **Télécharger le dataset** depuis Kaggle
2. **Placer le fichier CSV** dans le même dossier que le notebook
3. **Exécuter toutes les cellules** dans l'ordre

### 3. Cellules du Notebook

| Cellule | Description |
|---------|-------------|
| 1 | Imports des librairies |
| 2 | Chargement des données |
| 3 | Nettoyage simple |
| 4 | Sélection des features |
| 5 | Encodage des variables catégorielles |
| 6 | Split train/test |
| 7 | Entraînement du modèle |
| 8 | Évaluation |
| 9 | Visualisations |
| 10 | Sauvegarde |
| 11 | Fonction de prédiction |
| 12 | Test de prédiction |
| 13 | Résumé |

## 🔧 Features du Modèle

Le modèle utilise 7 features principales :

1. **neighbourhood_group** : Borough (Manhattan, Brooklyn, etc.)
2. **room_type** : Type de logement (Entire home, Private room, etc.)
3. **latitude** : Latitude
4. **longitude** : Longitude
5. **minimum_nights** : Nombre minimum de nuits
6. **number_of_reviews** : Nombre d'avis
7. **availability_365** : Disponibilité annuelle

## 📈 Performance

- **R² (Test)** : ~0.445
- **MAE** : ~$50-60
- **Temps d'entraînement** : < 30 secondes
- **Temps de prédiction** : < 1ms

## 🚀 Prochaines Étapes

1. Créer un script `predict.py` pour les prédictions
2. Écrire des tests unitaires
3. Implémenter une API REST
4. Configurer GitHub Actions
5. Ajouter une interface web (optionnel)

## 📝 Notes

- Le modèle filtre les prix > $1000 pour éviter les outliers
- Les variables catégorielles sont encodées avec LabelEncoder
- Gradient Boosting est choisi pour son bon équilibre simplicité/performance

## 🤝 Contribution

Pour contribuer à ce projet :
1. Fork le repository
2. Créez une nouvelle branche
3. Commitez vos changes
4. Poussez vers la branche
5. Créez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT.
