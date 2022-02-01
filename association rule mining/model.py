import pickle
import pandas as pd

class Recommender():


    def __init__(self):
        movies_df = pd.read_csv('../Data/movies.csv')
        ratings_df = pd.read_csv('../Data/ratings.csv')
        title_mapping = pd.Series(data=movies_df.title.values, index=movies_df.movieId.values)
        posters_df = pd.read_csv('../Data/movie_poster.csv', delimiter=';')

        with open('association-rules.pickle','rb') as file:
            ar = pickle.load(file)
            self.ar = ar
        with open('predictions.pickle','rb') as file:
            preds_df = pickle.load(file)
            self.preds_df = preds_df

        posters_mapping = pd.Series(posters_df.poster.values, index=posters_df.title.values)

        self.posters = posters_mapping
        self.title_mapping = title_mapping
        self.movies_df = movies_df
        self.ratings_df = ratings_df

    def recommend_movies(self, userId , num_recommendations=10):
        # Get and sort the user's predictions
        user_row_number = userId - 1  # UserID starts at 1, not 0
        sorted_user_predictions = self.preds_df.iloc[user_row_number].sort_values(ascending=False)  # UserID starts at 1

        # Get the user's data and merge in the movie information.
        user_data = self.ratings_df[self.ratings_df.userId == (userId)]
        user_full = (user_data.merge(self.movies_df, how='left', left_on='movieId', right_on='movieId').
                     sort_values(['rating'], ascending=False)
                     )

        print('User {0} has already rated {1} movies.'.format(userId, user_full.shape[0]))
        print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations = (self.movies_df[~self.movies_df['movieId'].isin(user_full['movieId'])].
                               merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                     left_on='movieId',
                                     right_on='movieId').
                               rename(columns={user_row_number: 'Predictions'}).
                               sort_values('Predictions', ascending=False).
                               iloc[:num_recommendations, :-1]
                               )

        return user_full, recommendations


    def get_explanations(self, userId, n_rules=10):

        already_rated, predictions = self.recommend_movies(userId)
        seen = list(already_rated.movieId.values)
        alle = list(self.movies_df.movieId.values)
        unseen = [x for x in alle if x not in seen]

        # filter the rules
        filtered_rules = self.ar.loc[
            self.ar['consequents'].apply(lambda y: True if y in unseen else False)]
        filtered_rules = filtered_rules.loc[filtered_rules['antecedents'].apply(lambda x: True if x in seen else False)]
        filtered_rules = filtered_rules.sort_values(by='lift', ascending=False)
        antecedents = filtered_rules[:n_rules].antecedents
        consequents = filtered_rules[:n_rules].consequents

        filtered_rules['antecedents_t'] = filtered_rules['antecedents'].apply(lambda x: self.title_mapping[x])
        filtered_rules['consequents_t'] = filtered_rules['consequents'].apply(lambda x: self.title_mapping[x])
        return filtered_rules