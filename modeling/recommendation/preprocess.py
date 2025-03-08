import pandas as pd
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

def basic_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Basic transformation of the MovieLens dataset"""
    # ----------------------------------------------------
    # Note:
    # ----------------------------------------------------
    # user ids are ordered sequentially from 1..138493
    # with no missing numbers
    # movie ids are integers from 1..131262
    # NOT all movie ids appear
    # there are only 26744 movie ids
    # write code to check it yourself!
    # ----------------------------------------------------
    # make the user ids go from 0...N-1
    df['userId'] = df['userId'] - 1
    # create a mapping for movie ids
    unique_movie_ids = set(df['movieId'].values)
    movie2idx: dict = {}
    count = 0
    for movie_id in unique_movie_ids:
        movie2idx[movie_id] = count
        count += 1
    # add them to the data frame
    # takes awhile
    df['movie_idx'] = df.apply(lambda row: movie2idx[row['movieId']], axis=1)
    df = df.drop(columns=['timestamp'])
    return df

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the MovieLens dataset into train and test sets"""
    # load in the data
    N = df['userId'].max() + 1 # number of users
    M = df['movie_idx'].max() + 1 # number of movies
    # split into train and test
    df = shuffle(df)
    cutoff = int(0.8*len(df))
    df_train = df.iloc[:cutoff]
    df_test = df.iloc[cutoff:]
    return df_train, df_test

def convert_data_to_dict(df: pd.DataFrame, subset: str) -> tuple[dict, dict, dict]:
    """Convert the MovieLens dataset to JSON format"""
    # a dictionary to tell us which users have rated which movies
    user2movie = {}
    # a dictionary to tell us which movies have been rated by which users
    movie2user = {}
    # a dictionary to look up ratings
    usermovie2rating = {}
    print(f"Calling: update_dictionaries for subset {subset}")
    count = 0
    def update_dictionaries(row):
        nonlocal count
        count += 1
        if count % 100000 == 0:
            print("processed: %.3f" % (float(count)/len(df)))
        i = int(row.userId)
        j = int(row.movie_idx)
        user2movie[i] += user2movie.get(i, [j])
        movie2user[j] += movie2user.get(j, [i])
        usermovie2rating[(i,j)] = row.rating
    df.apply(update_dictionaries, axis=1)
    return user2movie, movie2user, usermovie2rating

def save_as_sparse_data(df: pd.DataFrame, subset: str):
    """Convert the MovieLens dataset to sparse matrix format"""
    N = df['userId'].max() + 1 # number of users
    M = df['movie_idx'].max() + 1 # number of movies
    A = lil_matrix((N, M))
    print(f"Calling: update_{subset}")
    count = 0
    def update_data(row):
        nonlocal count
        count += 1
        if count % 100000 == 0:
            print(f"processed: {float(count)/len(df)}.3f")
        i = int(row.userId)
        j = int(row.movie_idx)
        A[i,j] = row.rating
    df.apply(update_data, axis=1)
    # mask, to tell us which entries exist and which do not
    A = A.tocsr()
    mask = (A > 0)
    save_npz(f"A{subset}.npz", A)