#!/usr/bin/env python3
"""
Dataset: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
"""

from pathlib import Path
import kaggle
import shutil

def check_kaggle_setup():
    try:
        #try to authenticate with the kaggle.json file
        kaggle.api.authenticate()
        return True
    except Exception as e:
        error_message = f'Kaggle API setup issue: {e}\nTo set up Kaggle API:\n 1. Go to https://www.kaggle.com/account\n 2. Click "Create New API Token" to download kaggle.json\n 3. Place kaggle.json in: ~/.kaggle/'
        print(error_message)
        return False

def download_movies_dataset(input_dir):
    if not check_kaggle_setup():
        return False 
    
    try:
        print("Downloading The Movies Dataset from Kaggle...")
        
        #download the dataset
        kaggle.api.dataset_download_files(
            'rounakbanik/the-movies-dataset',
            path=input_dir,
            unzip=True
        )
        
        print(f"Dataset downloaded successfully to {input_dir}")
        
        #list all the downloaded files and their sizes
        print("\nDownloaded files:")
        dir_path = Path(input_dir)
        for file in dir_path.iterdir():
            size_in_mb = file.stat().st_size / (1024 * 1024)
            print(f'{file.name} Size: {size_in_mb} MB')

        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def main():
    #the project root is 2 levels up from this file
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / 'data'
    
    #delete the input directory if it already exists
    if input_dir.exists():
            print(f"Removing existing input directory: {input_dir}")
            shutil.rmtree(input_dir) #recursively remove the directory and everything in the directory
    
    #create the input directory
    input_dir.mkdir()
    
    success = download_movies_dataset(input_dir)
    
    if success:
        print("\nDataset download completed successfully!")
    else:
        print("\nDataset download failed!")
    

if __name__ == "__main__":
    main()
