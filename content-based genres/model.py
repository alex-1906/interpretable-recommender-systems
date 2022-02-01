import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class Recommender:

    def __init__(self):
        '''Loading the required Datasets from the Data Folder'''
        metadata_df = pd.read_csv('../Data/movies_metadata.csv', low_memory=False)
        ratings_df = pd.read_csv('../Data/ratings.csv')
        movies_df = pd.read_csv('../Data/movies.csv')
        posters_df = pd.read_csv('../Data/movie_poster.csv', delimiter=';')
        links_df = pd.read_csv('../Data/links.csv')

        '''Preprocessing'''
        movies_df['genres'] = movies_df.genres.str.split('|')
        movies_df.drop_duplicates(subset='title', inplace=True, ignore_index=True)

        # One-hot encoding of the categories
        # First let's make a copy of the movies_df
        movies_with_genres = movies_df.copy(deep=True)

        # Let's iterate through movies_df, then append the movie genres as columns of 1s or 0s.
        # 1 if that column contains movies in the genre at the present index and 0 if not.

        for index, row in movies_df.iterrows():
            for genre in row['genres']:
                movies_with_genres.at[index, genre] = 1
        movies_with_genres.drop(columns='(no genres listed)',inplace=True)
        movies_with_genres.fillna(0, inplace=True)

        ratings_df.drop('timestamp', axis=1, inplace=True)
        movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')
        movie_ratings_user = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

        indices = pd.Series(movies_df.index, index=movies_df['title'])

        posters = pd.Series(posters_df['poster'].values, index =posters_df['title'])

        self.ratings_df = ratings_df
        self.movie_ratings_user = movie_ratings_user
        self.indices = indices
        self.movies_with_genres = movies_with_genres
        self.movies_df = movies_df
        self.posters = posters

    def get_user_profile(self,userId):
        user_ratings = self.movie_ratings_user.iloc[userId]
        user_ratings.dropna(inplace=True)
        liste = []
        for i in user_ratings.index:
            liste.append(self.indices[i])
        user_profile_df = self.movies_with_genres.iloc[liste]
        # user_profile_df.drop(columns=['movieId','title','genres'],inplace=True)
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
        preferences = np.array(liste)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_ratings = scaler.fit_transform(preferences.reshape(-1, 1))
        liste = []
        for i in scaled_ratings:
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