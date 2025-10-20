from download_data import download_movies_dataset
from pathlib import Path
from data_cleaning import MoviesPreprocessor

def main():
    project_root = Path(__file__).parent.parent
    input_dir = project_root / 'data'
    
    download_movies_dataset(input_dir)
    preprocessor = MoviesPreprocessor()
    preprocessor.execute_preprocessing_pipeline()


if __name__ == "__main__":
    main()
