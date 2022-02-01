import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:

    def __init__(self):
        '''Loading the required Datasets from the Data Folder'''
        metadata_df = pd.read_csv('../Data/movies_metadata.csv', low_memory=False)
        ratings_df = pd.read_csv('../Data/ratings.csv')
        movies_df = pd.read_csv('../Data/movies.csv')
        posters_df = pd.read_csv('../Data/movie_poster.csv', delimiter=';')
        links_df = pd.read_csv('../Data/links.csv')

        '''Preprocessing'''

        # Preprocessing
        movies_df['genres'] = movies_df.genres.str.split('|')
        movies_with_genres = movies_df.copy(deep=True)
        movies_df.drop_duplicates(subset='title', inplace=True, ignore_index=True)
        # One-hot encoding the categories
        for index, row in movies_df.iterrows():
            for genre in row['genres']:
                movies_with_genres.at[index, genre] = 1

        movies_with_genres = movies_with_genres.fillna(0)

        # Let's look at the ratings dataframe
        ratings_df.drop('timestamp', axis=1, inplace=True)
        movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')
        movie_ratings_user = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

        indices = pd.Series(movies_df.index, index=movies_df['title'])

        #get movie posters
        posters_mapping = pd.Series(posters_df.poster.values, index=posters_df.title.values)



        self.metadata = metadata_df
        self.ratings = ratings_df
        self.movies = movies_df
        self.posters = posters_mapping
        self.movie_ratings_user = movie_ratings_user
        self.movies_with_genres = movies_with_genres
        self.movies_df = movies_df
        self.indices = pd.Series(index=movies_df.title.values,data=movies_df.index)

    def get_user_profile(self,userId):
        user_ratings = self.movie_ratings_user.iloc[userId]
        user_ratings.dropna(inplace=True)
        liste = []
        for i in user_ratings.index:
            liste.append(self.indices[i])
        user_profile_df = self.movies_with_genres.iloc[liste]
        return user_ratings, user_profile_df

    def get_user_preferences(self,userId):
        user_ratings, user_profile = self.get_user_profile(userId)
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

    def get_user_recommendations(self,userId):
        movies = self.movies_with_genres.copy(deep=True)
        movies.drop(columns=['movieId', 'title', 'genres'], inplace=True)

        user_preferences = self.get_user_preferences(userId)
        user_preferences.drop(columns='genre', inplace=True)

        movies_array = movies.to_numpy()
        preferences_array = user_preferences.to_numpy()

        scores = movies_array.dot(preferences_array)
        liste = []
        for i in scores:
            liste.append(i[0])
        recommendations = pd.DataFrame(list(zip(self.movies_df.title.values, liste)), columns=['title', 'score'])
        #  recommendations.score = recommendations.score/recommendations.score.max()
        recommendations.sort_values(by='score', ascending=False, inplace=True)
        return recommendations

    def get_influences(self,movieId, userId):
        ratings, profile = self.get_user_profile(userId)
        ratings_df = pd.DataFrame(list(zip(ratings.index, ratings.values)), columns=['title', 'rating'])
        titles = ratings_df.title.values
        ratings_df = ratings_df.drop(columns='title')
        profile_df = profile.reset_index()
        profile_df = profile_df.drop(columns=['index', 'movieId', 'title', 'genres'])

        liste = []
        for i in ratings.index:
            liste.append(self.indices[i])
        movies = self.movies_with_genres.copy(deep=True)
        movies.drop(columns=['movieId', 'title', 'genres'], inplace=True)
        #   movies.drop(liste,inplace=True)

        old_preferences = profile_df.T.dot(ratings_df)
        old_recommendations = self.get_user_recommendations(userId)
        influences = []
        for j in ratings_df.index:
            r = ratings_df.drop(index=j)
            p = profile_df.drop(index=j)
            new_pref = p.T.dot(r)
            recommendations = movies.dot(new_pref)
            recommendations['title'] = self.movies_with_genres.title
            influences.append(abs(recommendations.loc[movieId].rating - old_recommendations.loc[movieId].score))
        influences_df = pd.DataFrame(list(zip(titles, influences)), columns=['titles', 'influences'])
        influences_df.sort_values(by='influences', ascending=False, inplace=True)
        influences_df.influences = influences_df.influences / influences_df.influences.max()
        return influences_df