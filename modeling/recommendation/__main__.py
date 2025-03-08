import click
import kaggle
import zipfile
import pandas as pd
from pathlib import Path
from modeling.config import logger
from kaggle.api.kaggle_api_extended import KaggleApi

@click.group("recommendation")
def cli():
    """MovieLens Recommendation System CLI"""
    pass

@cli.command()
@click.option('--data-dir', default='./.local/large_files/movielens-20m-dataset', help='Directory to store the dataset')
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