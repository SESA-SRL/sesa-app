import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger les modèles
model_dict = {
    "Chiffre d'affaires (SESA %)": joblib.load("modele_SESA_optimise.pkl"),
    "Productivité": joblib.load("model_productivity.pkl"),
    "Satisfaction (score sur 10)": joblib.load("model_satisfaction.pkl"),
    "Marge (Part CA nouveaux produits %)": joblib.load("model_marge.pkl")
}

# Charger les colonnes d'entrée
colonnes = joblib.load("colonnes_utiles.pkl")

st.title("📊 Simulation de performance globale de SESA SRL")
st.markdown("Ajustez les variables clés pour estimer automatiquement les impacts sur tous les indicateurs stratégiques.")

# Interface sliders
donnees = {}
for col in colonnes:
    donnees[col] = st.slider(col, 0.0, 100.0, 50.0)

df_input = pd.DataFrame([donnees])

# Prédictions simultanées
if st.button("📌 Prédire toutes les performances estimées"):
    for nom, modele in model_dict.items():
        try:
            df_input_filtre = df_input[modele.feature_names_in_]
            prediction = modele.predict(df_input_filtre)[0]
            st.success(f"✅ **{nom}** estimé : **{prediction:.2f}**")
        except Exception as e:
            st.error(f"❌ Erreur sur {nom} : {e}")
