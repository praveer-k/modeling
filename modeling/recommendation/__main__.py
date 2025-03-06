import os
import kagglehub
import pandas as pd

from pathlib import Path
from click import command, option, group
from builtins import range, input

@group("recommendation")
def cli():
    """MovieLens Recommendation System CLI"""
    pass

@cli.command()
@option('--data-dir', 
        default='./.local/large_files/movielens-20m-dataset',
        help='Directory to store the dataset')
@option('--force/--no-force', 
        default=False,
        help='Force download even if files exist')
def download(data_dir: str, force: bool):
    """Download the MovieLens 20M dataset"""
    try:
        kagglehub.init(oauth=True)
        ratings_file = kagglehub.load('grouplens/movielens-20m-dataset/rating.csv')
        df = pd.read_csv(ratings_file)
        os.makedirs(data_dir, exist_ok=True)
        df.to_parquet(os.path.join(data_dir, 'rating.parquet'))
        print(f"Dataset downloaded to {data_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# @cli.command()
# @option('--data-dir', 
#         default='../large_files/movielens-20m-dataset',
#         help='Directory containing the dataset')
# @option('--output-file',
#         default='edited_rating.csv',
#         help='Output file name for preprocessed data')
# def preprocess(data_dir: str, output_file: str):
#     """Preprocess the MovieLens dataset"""
#     from .preprocess import preprocess_data
#     data_dir = Path(data_dir)
#     preprocess_data(data_dir, output_file)

# @cli.command()
# @option('--data-file', 
#         default='../large_files/movielens-20m-dataset/edited_rating.csv',
#         help='Path to preprocessed data file')
# @option('--factors', 
#         default=43,
#         help='Number of latent factors')
# @option('--epochs', 
#         default=20,
#         help='Number of training epochs')
# @option('--reg', 
#         default=0.01,
#         help='Regularization parameter')
# def train(data_file: str, factors: int, epochs: int, reg: float):
#     """Train matrix factorization model"""
#     from .matrix_factorization import train_model
#     train_model(data_file, factors, epochs, reg)

if __name__ == "__main__":
    cli()