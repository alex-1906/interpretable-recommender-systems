import pandas as pd
import numpy as np
import streamlit as st

ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

#Preprocessing
movies_df['genres'] = movies_df.genres.str.split('|')
movies_with_genres = movies_df.copy(deep=True)
movies_df.drop_duplicates(subset='title',inplace=True,ignore_index=True)
#One-hot encoding the categories
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movies_with_genres.at[index, genre] = 1

movies_with_genres = movies_with_genres.fillna(0)

#Let's look at the ratings dataframe
ratings_df.drop('timestamp',axis=1,inplace=True)
movie_ratings = pd.merge(ratings_df,movies_df,on='movieId')
movie_ratings_user = movie_ratings.pivot_table(index='userId',columns='title',values='rating')

indices = pd.Series(movies_df.index,index=movies_df['title'])


def get_user_profile(userId):
    user_ratings = movie_ratings_user.iloc[userId]
    user_ratings.dropna(inplace = True)
    liste = []
    for i in user_ratings.index:
        liste.append(indices[i])
    user_profile_df = movies_with_genres.iloc[liste]
    return user_ratings,user_profile_df


def get_user_preferences(userId):
    user_ratings, user_profile = get_user_profile(userId)
    user_profile = user_profile.drop(columns=['movieId', 'title', 'genres'])
    categories = user_profile.columns

    ratings = user_ratings.to_numpy().reshape(-1, 1)
    profile = user_profile.to_numpy()

    user_preferences = profile.T.dot(ratings)
    liste = []
    for i in user_preferences:
        liste.append(i[0])
    user_preferences = pd.DataFrame(list(zip(categories, liste)), columns=['genre', 'preference'])
    return user_preferences


def get_user_recommendations(userId):
    movies = movies_with_genres.copy(deep=True)
    movies.drop(columns=['movieId', 'title', 'genres'], inplace=True)

    user_preferences = get_user_preferences(userId)
    user_preferences.drop(columns='genre', inplace=True)

    movies_array = movies.to_numpy()
    preferences_array = user_preferences.to_numpy()

    scores = movies_array.dot(preferences_array)
    liste = []
    for i in scores:
        liste.append(i[0])
    recommendations = pd.DataFrame(list(zip(movies_df.title.values, liste)), columns=['title', 'score'])
    #  recommendations.score = recommendations.score/recommendations.score.max()
    recommendations.sort_values(by='score', ascending=False, inplace=True)
    return recommendations

def get_influences(movieId,userId):
    ratings, profile = get_user_profile(userId)
    ratings_df = pd.DataFrame(list(zip(ratings.index, ratings.values)), columns=['title', 'rating'])
    titles = ratings_df.title.values
    ratings_df = ratings_df.drop(columns='title')
    profile_df = profile.reset_index()
    profile_df = profile_df.drop(columns=['index','movieId','title','genres'])

    liste = []
    for i in ratings.index:
        liste.append(indices[i])
    movies = movies_with_genres.copy(deep=True)
    movies.drop(columns=['movieId','title','genres'],inplace=True)
 #   movies.drop(liste,inplace=True)

    old_preferences = profile_df.T.dot(ratings_df)
    old_recommendations = get_user_recommendations(userId)
    influences = []
    for j in ratings_df.index:
        r = ratings_df.drop(index=j)
        p = profile_df.drop(index=j)
        new_pref = p.T.dot(r)
        recommendations = movies.dot(new_pref)
        recommendations['title'] = movies_with_genres.title
    #    print(recommendations)
        influences.append(abs(recommendations.loc[movieId].rating-old_recommendations.loc[movieId].score))
    influences_df = pd.DataFrame(list(zip(titles,influences)),columns=['titles','influences'])
    influences_df.sort_values(by='influences',ascending=False,inplace=True)
    influences_df.influences = influences_df.influences/influences_df.influences.max()
    return influences_df


#for i in recommendations.index:
 #   print(get_influences(i,15))

st.title("Recommender System Dashboard")
def counter_reset():
    st.session_state.count = 0
selected_userId =  st.number_input('Choose user', min_value=1, max_value=610, value=1, step=1,on_change=counter_reset)
st.write('Preferences of user: ',selected_userId)
#get recommendations and standardize values between 0 and 1
recommendations = get_user_recommendations(selected_userId).head()
recommendations.score = recommendations.score/recommendations.score.max()


if 'count' not in st.session_state:
	st.session_state.count = 0


def increment_counter():
	st.session_state.count += 1



if(st.button('Get next recommendation',on_click=increment_counter)):
    st.write(st.session_state.count)
    st.write('Here is your ',st.session_state.count,'th recommendation: ', recommendations.iloc[st.session_state.count-1].title)
    st.write('The influences of the movies rated by you are the following: ')
    st.write(get_influences(st.session_state.count,selected_userId))


