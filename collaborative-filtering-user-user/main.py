from model import Recommender
import streamlit as st
import pandas as pd

recommender = Recommender()
posters = recommender.posters
#Methods for button controlling
def counter_reset():
    st.session_state.count = 0
if 'count' not in st.session_state:
	st.session_state.count = 0
def increment_counter():
	st.session_state.count += 1

st.title("Recommender System Dashboard: Collaborative filtering user-user")
selected_userId =  st.number_input('Choose user', min_value=1, max_value=610, value=1, step=1,on_change=counter_reset)
recommendations = recommender.get_neighbors_votes(selected_userId).head(5)

st.write('Recommendations for user: ',selected_userId)

if(st.button('Get next recommendation',on_click=increment_counter)):
    if(st.session_state.count >= 6):
        st.write('Choose next user')
    else:
        st.write('Here is your ',st.session_state.count,'th recommendation: ', recommendations.index[st.session_state.count-1])
        try:
            st.image(posters[recommendations.index[st.session_state.count-1]],width=100)
        except Exception:
            st.image("https://i.ibb.co/TqJsxKM/IMG-20190204-WA0001.jpg",width=100)
        row = recommendations.iloc[st.session_state.count-1]
        occurences = []
        for i in range(1, 6):
            occurences.append(row.root_ratings.count(str(i)))
        votes = pd.DataFrame(zip(['⭐','⭐⭐','⭐⭐⭐','⭐⭐⭐⭐','⭐⭐⭐⭐⭐'],occurences), columns=['             stars          ','#votes of 10 neighbors'])
        st.write(votes)