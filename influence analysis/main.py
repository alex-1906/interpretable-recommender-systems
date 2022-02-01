from model import Recommender
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


recommender = Recommender()
posters = recommender.posters
#Methods for button controlling
def counter_reset():
    st.session_state.count = 0
if 'count' not in st.session_state:
	st.session_state.count = 0
def increment_counter():
	st.session_state.count += 1

st.title("Recommender System Dashboard: Influence analysis")
selected_userId =  st.number_input('Choose user', min_value=1, max_value=610, value=1, step=1,on_change=counter_reset)
recommendations = recommender.get_user_recommendations(selected_userId).head()
st.write('Recommendations for user: ',selected_userId)


if(st.button('Get next recommendation',on_click=increment_counter)):
    if (st.session_state.count >= 6):
        st.write('Choose next user')
    else:
        st.write('Here is your ',st.session_state.count,'th recommendation: ')
        try:
            st.image(posters[recommendations.index[st.session_state.count-1]],width=200)
        except Exception:
            st.image("https://i.ibb.co/TqJsxKM/IMG-20190204-WA0001.jpg",width=200)
        st.write('Influence of your ratings: ')
        influences = recommender.get_influences(st.session_state.count,selected_userId).head()
        #plot the bar chart of influences
        fig = plt.figure()
        #figsize=(18, 5)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.barh(influences.titles, influences.influences,color='#00876c')

        st.pyplot(fig)