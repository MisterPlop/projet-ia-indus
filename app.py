import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from predict import load_model, predict_price, check_requirements
import json

# Configuration de la page
st.set_page_config(
    page_title="🏠 Prédicteur de Prix Airbnb",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF5A5F;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .subheader {
        text-align: center;
        color: #767676;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #FF5A5F 0%, #FF8E8F 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF5A5F;
        margin: 1rem 0;
        color: #FF5A5F;
        font-weight: 500;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_cache():
    """Charge le modèle avec mise en cache"""
    try:
        return load_model()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

@st.cache_data
def get_neighborhoods():
    """Récupère la liste des quartiers disponibles"""
    # Ces valeurs peuvent être ajustées selon votre modèle
    return ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']

@st.cache_data
def get_room_types():
    """Récupère les types de chambres disponibles"""
    return ['Entire home/apt', 'Private room', 'Shared room']

def main():
    # En-tête principal
    st.markdown('<div class="main-header">🏠 Prédicteur de Prix Airbnb</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Utilisez notre IA pour estimer le prix de votre logement</div>', unsafe_allow_html=True)
    
    # Vérification des requirements
    try:
        check_requirements()
    except SystemExit:
        st.error("❌ Certaines dépendances sont manquantes. Veuillez installer les packages requis.")
        st.stop()
    
    # Chargement du modèle
    model_data = load_model_cache()
    if model_data is None:
        st.error("❌ Impossible de charger le modèle. Vérifiez que le dossier 'pickles' existe et contient un modèle.")
        st.stop()
    
    # Interface utilisateur
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📍 Localisation")
        neighbourhood_group = st.selectbox(
            "Groupe de quartiers",
            get_neighborhoods(),
            help="Sélectionnez le quartier de votre logement"
        )
        
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.7589,
                step=0.0001,
                format="%.6f"
            )
        with col_lon:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-73.9851,
                step=0.0001,
                format="%.6f"
            )
        
        # Affichage de la carte
        map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_data, zoom=10)
    
    with col2:
        st.markdown("### 🏨 Propriétés du logement")
        room_type = st.selectbox(
            "Type de chambre",
            get_room_types(),
            help="Sélectionnez le type de votre logement"
        )
        
        minimum_nights = st.number_input(
            "Nombre minimum de nuits",
            min_value=1,
            value=2,
            help="Durée minimale de séjour requise"
        )
        
        number_of_reviews = st.number_input(
            "Nombre d'avis",
            min_value=0,
            value=10,
            help="Nombre total d'avis reçus"
        )
        
        availability_365 = st.slider(
            "Disponibilité (jours par an)",
            min_value=0,
            max_value=365,
            value=365,
            help="Nombre de jours disponibles à la location par an"
        )
    
    # Bouton de prédiction
    st.markdown("---")
    predict_button = st.button("🎯 Prédire le prix", type="primary", use_container_width=True)
    
    if predict_button:
        # Préparation des données
        listing_data = {
            'neighbourhood_group': neighbourhood_group,
            'room_type': room_type,
            'latitude': latitude,
            'longitude': longitude,
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'availability_365': availability_365
        }
        
        # Prédiction
        with st.spinner("Calcul de la prédiction..."):
            try:
                predicted_price = predict_price(model_data, listing_data)
                
                # Affichage du résultat
                st.markdown(f"""
                <div class="prediction-box">
                    💰 Prix prédit : ${predicted_price:.2f} par nuit
                </div>
                """, unsafe_allow_html=True)
                
                # Informations complémentaires
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prix mensuel estimé",
                        f"${predicted_price * 30:.2f}",
                        delta=f"{((predicted_price * 30) / 2000 - 1) * 100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Prix par semaine",
                        f"${predicted_price * 7:.2f}",
                        delta=f"{((predicted_price * 7) / 500 - 1) * 100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Revenus annuels",
                        f"${predicted_price * availability_365:.2f}",
                        delta=f"{availability_365} jours"
                    )
                
                # Graphique de comparaison
                st.markdown("### 📊 Analyse comparative")
                
                # Création de données de comparaison fictives pour la démonstration
                comparison_data = {
                    'Type': ['Votre logement', 'Moyenne quartier', 'Moyenne ville'],
                    'Prix': [predicted_price, predicted_price * 0.85, predicted_price * 0.9],
                    'Couleur': ['#FF5A5F', '#FF8E8F', '#FFB6B8']
                }
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=comparison_data['Type'],
                        y=comparison_data['Prix'],
                        marker_color=comparison_data['Couleur'],
                        text=[f"${price:.2f}" for price in comparison_data['Prix']],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Comparaison des prix",
                    yaxis_title="Prix par nuit ($)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Conseils d'optimisation
                st.markdown("### 💡 Conseils d'optimisation")
                
                tips = []
                if availability_365 < 300:
                    tips.append("🗓️ Augmentez la disponibilité pour maximiser vos revenus")
                if number_of_reviews < 10:
                    tips.append("⭐ Encouragez plus d'avis pour améliorer votre visibilité")
                if minimum_nights > 5:
                    tips.append("📅 Réduisez le minimum de nuits pour attirer plus de réservations")
                
                if tips:
                    for tip in tips:
                        st.markdown(f"""
                        <div class="info-box">
                            {tip}
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
    
    # Sidebar avec informations supplémentaires
    st.sidebar.markdown("### 📊 Informations sur le modèle")
    st.sidebar.info("""
    Notre modèle IA analyse plus de 7 variables clés pour prédire le prix optimal de votre logement Airbnb :
    
    🏠 **Type de logement**  
    📍 **Localisation géographique**  
    📅 **Disponibilité annuelle**  
    ⭐ **Nombre d'avis**  
    🗓️ **Minimum de nuits**  
    """)
    
    st.sidebar.markdown("### 🔧 Paramètres avancés")
    show_debug = st.sidebar.checkbox("Afficher les informations de debug")
    
    if show_debug and predict_button:
        st.sidebar.markdown("### 🐛 Debug Info")
        st.sidebar.json(listing_data)
        if model_data:
            st.sidebar.write("**Modèle chargé :** ✅")
            if isinstance(model_data, dict):
                st.sidebar.write(f"**Features :** {model_data.get('features', 'Non disponible')}")

if __name__ == "__main__":
    main()