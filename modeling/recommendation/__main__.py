import click
import kaggle
import pickle
import zipfile
import pandas as pd
from pathlib import Path
from modeling.config import logger
from kaggle.api.kaggle_api_extended import KaggleApi

from modeling.recommendation.matrix_factorization import plot_train_test_loss, train_for_recommendation
from modeling.recommendation.preprocess import basic_transform, convert_data_to_dict, save_as_sparse_data, split_train_test

@click.group("recommendation")
def cli():
    """MovieLens Recommendation System CLI"""
    pass

@cli.command()
@click.option('--data-dir', default='.local/large_files/movielens-20m-dataset', help='Directory to store the dataset')
@click.option('--force/--no-force', default=False, help='Force download even if files exist')
def download(data_dir: str, force: bool):
    """Download the MovieLens 20M dataset"""
    try:
        # Create directory structure
        data_dir: Path = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file paths
        ratings_csv = data_dir / 'rating.csv'
        ratings_parquet = data_dir / 'rating.parquet'
        zip_file = data_dir / 'rating.csv.zip'

        # Check if files exist
        if not force and ratings_parquet.exists():
            logger.info(f"Dataset already exists at {ratings_parquet}")
            return
        
        # Download dataset
        logger.info(f"Downloading dataset to {ratings_csv}")
        api = KaggleApi()
        api.authenticate()
        kaggle.api.dataset_download_file(
            dataset='grouplens/movielens-20m-dataset',
            file_name='rating.csv',
            path=data_dir.as_posix(),
        ) 
        # Extract zip file
        logger.info(f"Extracting {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Convert to parquet
        if ratings_csv.exists():
            logger.info("Converting to parquet format...")
            df = pd.read_csv(ratings_csv)
            df.to_parquet(ratings_parquet)
            # Cleanup temporary files
            ratings_csv.unlink()  # Remove CSV
            zip_file.unlink()     # Remove ZIP
            logger.info(f"Dataset saved to {ratings_parquet}")
        else:
            raise FileNotFoundError(f"Failed to extract dataset to {ratings_csv}")
            
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

@cli.command()
@click.option('--data-dir', default='.local/large_files/movielens-20m-dataset', help='Directory containing the dataset')
@click.option('--output-file', default='edited_rating.csv', help='Output file name for preprocessed data')
def preprocess(data_dir: str, output_file: str):
    """Preprocess the MovieLens dataset"""
    data_dir: Path = Path(data_dir)
    output_file: Path = data_dir / output_file
    ratings_parquet: Path = data_dir / 'rating.parquet'
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        return
    df = pd.read_parquet(ratings_parquet.as_posix())
    df = basic_transform(df)
    df.to_parquet(output_file.as_posix(), index=False)
    df_train, df_test = split_train_test(df)
    user2movie, movie2user, usermovie2rating = convert_data_to_dict(df_train)
    _, _, usermovie2rating_test = convert_data_to_dict(df_test)
    # note: these are not really JSONs
    with open('user2movie.json', 'wb') as f:
        pickle.dump(user2movie, f)
    with open('movie2user.json', 'wb') as f:
        pickle.dump(movie2user, f)
    with open('usermovie2rating.json', 'wb') as f:
        pickle.dump(usermovie2rating, f)
    with open('usermovie2rating_test.json', 'wb') as f:
        pickle.dump(usermovie2rating_test, f)
    # save as sparse data
    save_as_sparse_data(df_train)
    save_as_sparse_data(df_test)

@cli.command()
@click.option('--data-dir', default='.local/large_files/movielens-20m-dataset', help='Directory containing the dataset')
@click.option('--plot', default=False, help='Output plot for train and test')
def train(data_dir: str, plot: bool):
    data_dir: Path = Path(data_dir)
    train_losses, test_losses = train_for_recommendation(data_dir)
    if plot:
        plot_train_test_loss(train_losses, test_losses)


if __name__ == "__main__":
    cli()