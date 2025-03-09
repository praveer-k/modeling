import pickle
from pathlib import Path
from typing import Tuple


def load_data(data_dir: Path = '.local/large_files/movielens-20m-dataset') -> Tuple[dict, dict, dict, dict]:
    user2movie_file = data_dir / 'user2movie.pickle'
    movie2user_file = data_dir / 'movie2user.pickle'
    usermovie2rating_file = data_dir / 'usermovie2rating.pickle'
    usermovie2rating_test_file = data_dir / 'usermovie2rating_test.pickle'
    if not user2movie_file.exists() or not movie2user_file.exists() or not usermovie2rating_file.exists() or not usermovie2rating_test_file.exists():
        raise FileNotFoundError("All files - user2movie.pickle, movie2user.pickle, usermovie2rating.pickle, usermovie2rating_test.pickle must exists. Try running preprocessing step before this step.")
    with open(user2movie_file, 'rb') as f:
        user2movie = pickle.load(f)
    with open(movie2user_file, 'rb') as f:
        movie2user = pickle.load(f)
    with open(usermovie2rating_file, 'rb') as f:
        usermovie2rating = pickle.load(f)
    with open(usermovie2rating_test_file, 'rb') as f:
        usermovie2rating_test = pickle.load(f)
    return user2movie, movie2user, usermovie2rating, usermovie2rating_test