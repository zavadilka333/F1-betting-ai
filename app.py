
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="F1 Betting Assistant (Demo)", layout="wide")

st.title("F1 Betting Assistant — Demo (Synthetic Data)")

st.markdown("""
This demo uses a synthetic dataset and a simple Random Forest model to predict the probability
that a driver finishes in TOP 3. Replace the dataset loading with real Ergast/FastF1 data for production use.
""")

@st.cache_data
def load_data(path="/mnt/data/sample_f1_data.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path="/mnt/data/model.pkl"):
    return joblib.load(path)

df = load_data()
model = load_model()

st.sidebar.header("Select race")
seasons = sorted(df['season'].unique(), reverse=True)
season = st.sidebar.selectbox("Season", seasons)
rounds = sorted(df[df['season']==season]['round'].unique())
rnd = st.sidebar.selectbox("Round", rounds)

race_df = df[(df['season']==season) & (df['round']==rnd)].copy()
st.write(f"### Race preview — Season {season} Round {rnd} — {race_df['track'].iloc[0]}")
st.dataframe(race_df[['driver','team','grid_pos','qualifying_lap_time','weather','odds']])

st.write("### Predicted probabilities for TOP3")
X_features = race_df[['grid_pos','qualifying_lap_time','race_incidents','mechanical_failures','team_strength','odds']]
probs = model.predict_proba(X_features)[:,1]
race_df['prob_top3'] = probs
race_df = race_df.sort_values('prob_top3', ascending=False).reset_index(drop=True)
st.dataframe(race_df[['driver','team','grid_pos','prob_top3','odds']])

st.write("### Suggested value bets (simple rule)")
# value = (probability * odds) - 1  -> positive suggests value
race_df['value'] = (race_df['prob_top3'] * race_df['odds']) - 1
value_bets = race_df[race_df['value'] > 0].sort_values('value', ascending=False)
if value_bets.empty:
    st.write("No value bets found (according to the simple rule).")
else:
    st.dataframe(value_bets[['driver','team','prob_top3','odds','value']])
