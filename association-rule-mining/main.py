import matplotlib.pyplot as plt
from model import Recommender
import streamlit as st

recommender = Recommender()
posters = recommender.posters

st.title("Recommender System Dashboard: Association rule mining")
selected_userId = st.number_input('Choose user', min_value=1, max_value=610, value=1, step=1)
st.write('Recommendations for user: ',selected_userId)
ar = recommender.get_explanations(selected_userId)
recommendations = recommender.get_explanations(selected_userId)
recommendations.drop_duplicates(subset='consequents',inplace=True)
recommendations.reset_index(inplace=True)

st.write(recommendations[['antecedents_t','consequents_t','support','lift']].head(10))
#for i in range(0,5):
 #   st.write(f"{recommendations.iloc[i].antecedents_t} => {recommendations.iloc[i].consequents_t}")
