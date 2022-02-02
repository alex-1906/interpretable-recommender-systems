import numpy as np
import pandas as pd
import datetime
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.sparse.linalg import svds


movies_df = pd.read_csv('../Data/movies.csv')
ratings_df = pd.read_csv('../Data/ratings.csv')

R_df = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
R = R_df.to_numpy()

#standardize the matrix R
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

#matrix factorization
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)


def recommend_movies(userId, num_recommendations=10):
    # Get and sort the user's predictions
    user_row_number = userId - 1  # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)  # UserID starts at 1

    # Get the user's data and merge in the movie information.
    user_data = ratings_df[ratings_df.userId == (userId)]
    user_full = (user_data.merge(movies_df, how='left', left_on='movieId', right_on='movieId').
                 sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userId, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='movieId',
                                 right_on='movieId').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )

    return user_full, recommendations

item_sets = []
top_n_start_time = datetime.datetime.now()
for userId in range(1, 610):
    already_rated, predictions = recommend_movies(userId)
    item_sets.append(predictions.movieId.values.tolist())
top_n_end_time = datetime.datetime.now()
print("time: ", top_n_end_time - top_n_start_time)
top_n_items = item_sets

te = TransactionEncoder()
te_ary = te.fit(top_n_items).transform(top_n_items, sparse=True)
topn_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
topn_df.columns = [str(i) for i in topn_df.columns]

apriori_start_time = datetime.datetime.now()
frequent_itemsets = apriori(topn_df, min_support=0.002, verbose=1, low_memory=True, use_colnames=True)
apriori_end_time = datetime.datetime.now()
print("Training duration: " + str(apriori_end_time - apriori_start_time))

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
rules = association_rules(frequent_itemsets)

#filter the rules
rules['consequents_length'] = rules['consequents'].apply(lambda x: len(x))
rules['antecedents_length'] = rules['antecedents'].apply(lambda x: len(x))
filtered_rules = rules[
    (rules['confidence'] > 0.3) & (rules['antecedents_length'] < 4) & (rules['consequents_length'] == 1)]
filtered_rules['consequents'] = filtered_rules.consequents.apply(lambda x: list(x))
filtered_rules['antecedents'] = filtered_rules.antecedents.apply(lambda x: list(x))
filtered_rules['consequents'] = filtered_rules.consequents.apply(lambda x: int(x[0]))
filtered_rules['antecedents'] = filtered_rules.antecedents.apply(lambda x: int(x[0]))


with open('association-rules.pickle','wb+') as file:
    pickle.dump(filtered_rules,file)

with open('predictions.pickle','wb+') as file:
    pickle.dump(preds_df,file)