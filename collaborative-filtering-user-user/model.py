import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:

    def __init__(self):
        '''Loading the required Datasets from the Data Folder'''
        ratings_df = pd.read_csv('../Data/ratings.csv')
        movies_df = pd.read_csv('../Data/movies.csv')
        posters_df = pd.read_csv('../Data/movie_poster.csv', delimiter=';')

        movies_df.drop_duplicates(subset='title', inplace=True, ignore_index=True)
        movies_with_genres = movies_df.merge(ratings_df, on='movieId')
        movies_with_genres.drop(columns='timestamp', inplace=True)
        movie_ratings_user = movies_with_genres.pivot_table(index='userId', columns='title', values='rating')

        #reverse mapping from title to index
        indices = pd.Series(movies_df.index, index=movies_df['title'])

        #center the ratings around zero and fill missing values with zeros
        avg_ratings = movie_ratings_user.mean(axis=1)
        movie_ratings_user_centered = movie_ratings_user.sub(avg_ratings, axis=0)
        movie_ratings_user_centered.fillna(0, inplace=True)

        #calculate the cosine similarity between the users
        cosine_sim_matrix = cosine_similarity(movie_ratings_user_centered, movie_ratings_user_centered)
        similarities = pd.DataFrame(cosine_sim_matrix, index=movie_ratings_user_centered.index,columns=movie_ratings_user_centered.index)

        posters_mapping = pd.Series(posters_df.poster.values, index=posters_df.title.values)

        self.similarities = similarities
        self.movie_ratings_user = movie_ratings_user
        self.posters = posters_mapping
        self.indices = indices


    def get_KNN(self, userId, k=3):
        knn = self.similarities.loc[userId].sort_values(ascending=False)[1:k + 1]
        return knn

    def get_user_ratings(self,userId):
        #take the non filled matrix here, to be able to drop all NaNs and therefore extract only the watched movies
        user_ratings = self.movie_ratings_user.loc[userId]
        user_ratings.dropna(inplace=True)
        user_ratings.sort_values(ascending=False, inplace=True)
        return user_ratings

    def get_scaled_user_ratings(self,userId):
        ratings = self.get_user_ratings(userId)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_ratings = scaler.fit_transform(np.array(ratings.values).reshape(-1, 1))
        liste = []
        for i in scaled_ratings.tolist():
            liste.append(i[0])
        df = pd.DataFrame(liste, index=ratings.index, columns=['rating_scaled'])
        df['rating'] = ratings.values
        df['root_user'] = userId
        return df

    def get_neighbors_votes(self,userId):
        knn = self.get_KNN(userId, k=10)
        # Get movies of the kNN and put them into a dataframe
        liste = []
        for i in knn.index:
            user_ratings = self.get_scaled_user_ratings(i)
            user_ratings['similarity'] = knn[i]
            liste.append(user_ratings)
        df = pd.concat(liste)
        # Remove movies which the user has already seen
        df_out = pd.DataFrame(index=df.index.unique())

        df_out['score'] = 0
        df_out['roots'] = ''
        df_out['root_ratings'] = ''

        for row in df.iterrows():
            similarity = row[1].similarity
            scaled_rating = row[1].rating_scaled
            df_out.loc[row[0], 'score'] += similarity * scaled_rating
            df_out.loc[row[0], 'roots'] += str(int(row[1].root_user)) + '|'
            df_out.loc[row[0], 'root_ratings'] += str(int(row[1].rating)) + '|'

        df_out['roots'] = df_out.roots.str.split('|')
        df_out['root_ratings'] = df_out.root_ratings.str.split('|')
        for row in df_out.iterrows():
            row[1].roots.pop()
            row[1].root_ratings.pop()

        df_out.sort_values(by='score', ascending=False, inplace=True)
        # Remove movies which the user has already seen
        already_seen = self.get_user_ratings(10).index.tolist()
        df_out.drop(already_seen, inplace=True, errors='ignore')
        return df_out