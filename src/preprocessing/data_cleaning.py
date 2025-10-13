import pandas as pd
import numpy as np
import string
import os
import shutil
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm #used for the progress bar

#TODO: for similarity matrices, drop rows where similarity score is below a certain threshold in order to reduce the size of the matrices

class MoviesPreprocessor:
    def __init__(self):
        #Project directory is 2 levels up from this file
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / 'data'
        self.output_path = self.project_root / 'data_cleaned'
        
        #Delete the output directory if it already exists
        if self.output_path.exists():
            print(f"Removing existing output directory: {self.output_path}")
            shutil.rmtree(self.output_path) #recursively remove the directory and everything in the directory
        
        self.output_path.mkdir()
        
        self.stemmer = SnowballStemmer('english')
        
        self.movies = None
        self.keywords = None
        self.credits = None
        
        self.content_sims = None
        self.genres_sims = None
        self.crew_sims = None
        self.ratings_sims = None
        self.movie_ratings = None
        
    def load_movies_dataset(self):
        print("Loading movies dataset...")
        self.movies = pd.read_csv(self.data_path / 'movies_metadata.csv')

    def load_links_dataset(self):
        print("Loading links dataset...")
        self.links = pd.read_csv(self.data_path / 'links_small.csv')
        #Clean the links dataset
        self.links = self.links[self.links['tmdbId'].notnull()]
        self.links['tmdbId'] = self.links['tmdbId'].astype(int)

    def load_keywords_dataset(self):
        print("Loading keywords dataset...")
        self.keywords = pd.read_csv(self.data_path / 'keywords.csv')

    def load_credits_dataset(self):
        print("Loading credits dataset...")
        self.credits = pd.read_csv(self.data_path / 'credits.csv')

    def load_movie_ratings_dataset(self):
        print("Loading movie ratings dataset...")
        self.movie_ratings = pd.read_csv(self.data_path / 'ratings_small.csv')

    def load_all_datasets(self):
        print("Loading all datasets...")
        self.load_movies_dataset()
        self.load_links_dataset()
        self.load_keywords_dataset()
        self.load_credits_dataset()
        self.load_movie_ratings_dataset()
    
    def clean_movies(self):
        print("Cleaning movies dataset...")
        # Rename 'id' column to 'tmdb_id'
        self.movies = self.movies.rename(columns={'id': 'tmdb_id'})

        # Convert tmdb_id to numeric, coerce errors to NaN
        self.movies['tmdb_id'] = pd.to_numeric(self.movies['tmdb_id'], errors='coerce')

        # Drop rows with missing tmdb_id
        self.movies.dropna(subset=['tmdb_id'], inplace=True)

        # Convert tmdb_id to int
        self.movies['tmdb_id'] = self.movies['tmdb_id'].astype(int)

        #Drop movies that don't appear in the links dataset to shrink the data size (otherwise there isn't enough RAM to load all of the data into memory)
        self.movies = self.movies[self.movies['tmdb_id'].isin(self.links['tmdbId'])]

        # Drop duplicate entries based on imdb_id and tmdb_id
        self.movies.drop_duplicates(subset=['imdb_id', 'tmdb_id'], inplace=True)

        # Drop rows where vote_count, vote_average, title, or tmdb_id is missing
        self.movies.dropna(subset=['vote_count', 'vote_average', 'title', 'tmdb_id'], inplace=True)

        # Drop or rename columns as needed
        self.movies.drop(columns=['original_title'], inplace=True, errors='ignore')

        #Convert 'genres' json column to a list of strings
        self.movies['genres'] = self.movies['genres'].apply(lambda element: eval(element))
        self.movies['genres'] = self.movies['genres'].apply(lambda element: [d['name'] for d in element] if isinstance(element, list) else [])

        # Drop unnecessary columns (ignore if missing)
        self.movies.drop(
            columns=[
                'belongs_to_collection', 'budget', 'homepage', 'popularity', 'production_companies',
                'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages',
                'status', 'poster_path', 'video', 'original_language', 'adult', 'imdb_id'
            ],
            inplace=True, errors='ignore'
        )

        # Combine 'tagline' and 'overview' columns
        self.movies['overview'] = self.movies['tagline'].fillna('').str.cat(self.movies['overview'].fillna(''), sep=' ')
        self.movies.drop(columns=['tagline'], inplace=True, errors='ignore')

        # Turn the overview column into a list of stemmed words
        self.movies['overview'] = self.movies['overview'].str.split()
        self.movies['overview'] = self.movies['overview'].apply(lambda l: [self.stemmer.stem(word.strip(string.punctuation)) for word in l])

        # Calculate the imdb score
        min_votes = self.movies['vote_count'].quantile(0.90)
        overall_vote_average = self.movies['vote_average'].mean()
        imdb_score = (
            (self.movies['vote_count'] * self.movies['vote_average'] + min_votes * overall_vote_average) /
            (self.movies['vote_count'] + min_votes)
        )
        self.movies['imdb_score'] = imdb_score
        unqualified_movies = self.movies['vote_count'] < min_votes
        self.movies.loc[unqualified_movies, 'imdb_score'] = np.nan

        # Drop the vote_count and vote_average columns after calculating the imdb score
        self.movies.drop(columns=['vote_count', 'vote_average'], inplace=True, errors='ignore')

    def clean_keywords(self):
        print("Cleaning keywords dataset...")
        #Convert the keywords column from a json object to a list of dictionaries
        self.keywords['keywords'] = self.keywords['keywords'].apply(lambda element: eval(element))
        
        #Rename the 'id' column to 'tmdb_id'
        self.keywords = self.keywords.rename(columns={'id': 'tmdb_id'})

        #Drop movies in the keywords table whose tmdb_id don't appear in the movies dataset (to avoid primary key error)
        self.keywords = self.keywords[self.keywords['tmdb_id'].isin(self.movies['tmdb_id'])]

        #Turn the keywords column into a list of stemmed words
        self.keywords['keywords'] = self.keywords['keywords'].apply(lambda element: [d['name'] for d in element] if isinstance(element, list) else [])

        #Filter out the keywords that only appear once in the movie dataset (not useful for finding similar movies)
        all_keywords = self.keywords['keywords'].apply(lambda l: pd.Series(l)).stack().value_counts()
        unique_keywords = all_keywords[all_keywords == 1]
        self.keywords['keywords'] = self.keywords['keywords'].apply(lambda l: [element for element in l if element not in unique_keywords.index])

        #Stem each of the keywords"""
        self.keywords['keywords'] = self.keywords['keywords'].apply(lambda l: [self.stemmer.stem(word) for word in l])


    def clean_credits(self):
        print("Cleaning credits dataset...")
        #Convert the cast and crew columns from a json object to a list of dictionaries
        self.credits['cast'] = self.credits['cast'].apply(lambda element: eval(element))
        self.credits['crew'] = self.credits['crew'].apply(lambda element: eval(element))

        #Make a new column for the directors
        self.credits['director'] = self.credits['crew'].apply(lambda element: [d['name'] for d in element if 'director' in d['job'].lower()] if isinstance(element, list) else [])
        
        #Turn the director names into a single string not seperated by spaces (otherwise first and last name will be treated as different terms in TF-IDF vectorization)"""
        self.credits['director'] = self.credits['director'].apply(lambda l: [''.join(fullname.split()) for fullname in l])

        #Drop the directors that only appear once in the dataset (not useful for finding similar movies)
        director_name_counts = self.credits['director'].apply(lambda l: pd.Series(l)).stack().value_counts()
        unique_directors = director_name_counts[director_name_counts == 1]
        self.credits['director'] = self.credits['director'].apply(lambda l: [element for element in l if element not in unique_directors.index][0:3]) #Take the top 3 directors

        #Make a new column for the actors
        actors = self.credits['cast'].apply(lambda l: [d['name'] for d in l if 'character' in d.keys() and d['character'] != ''][0:10]) #take the top 10 actors
        actors = actors.apply(lambda l: [''.join(fullname.split()) for fullname in l])
        self.credits['actors'] = actors

        #Rename the 'id' column to 'tmdb_id'
        self.credits = self.credits.rename(columns={'id': 'tmdb_id'})

        #Drop movies in the credits table whose tmdb_id don't appear in the movies dataset (to avoid primary key error)
        self.credits = self.credits[self.credits['tmdb_id'].isin(self.movies['tmdb_id'])]


        #Director should be weighted twice as much as the actors
        self.credits['director'] = self.credits['director'] * 2
        self.credits.head()

    
    def clean_movie_ratings(self):
        print("Cleaning movie ratings dataset...")
        # Rename movieId to tmdb_id first
        self.movie_ratings.rename(columns={'movieId': 'tmdb_id'}, inplace=True)
        
        # Filter to only include movies that exist in the movies dataset
        self.movie_ratings = self.movie_ratings[self.movie_ratings['tmdb_id'].isin(self.movies['tmdb_id'])]
        
        # Create pivot table
        self.movie_ratings = self.movie_ratings.pivot_table(index='userId', columns='tmdb_id', values='rating')


    def merge_datasets(self):
        print("Merging datasets...")
        self.movies = self.movies.merge(self.keywords, on = 'tmdb_id', how = 'inner')  
        self.movies = self.movies.merge(self.credits.loc[:, ['tmdb_id', 'actors', 'director']], on='tmdb_id', how='left')
        
        #If the left join is not perfect, the actors and director columns will be filled with an empty list
        self.movies['actors'] = self.movies['actors'].apply(lambda x: x if isinstance(x, list) else [])
        self.movies['director'] = self.movies['director'].apply(lambda x: x if isinstance(x, list) else [])


    def create_movie_genre_similarity_matrix(self):
        print("Creating movie genre similarity matrix...")
        #Turn the entries of the genres column into a single string, where individual genres are seperated by spaces
        genres = self.movies['genres'].apply(lambda l: ' '.join(l) if isinstance(l, list) else '')

        #Drop the genres column in the movies dataset
        self.movies.drop(columns = ['genres'], inplace = True)

        #Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1, 2), min_df = 0.01)
        dtm = tfidf_vectorizer.fit_transform(genres)

        #Create a similarity matrix
        genres_sim_matrix = pd.DataFrame(cosine_similarity(dtm, dense_output = True))
        genres_sim_matrix.index = self.movies['tmdb_id'].values
        genres_sim_matrix.columns = self.movies['tmdb_id'].values

        #Reshape the similarity matrix to have columns (movie1_id, movie2_id, similarity_score)
        genres_sims = genres_sim_matrix.stack().reset_index()
        genres_sims.columns = ['movie1_id', 'movie2_id', 'similarity_score']

        #Drop entries where similarity is 0 or similarity is calculated between the same movies
        self.genres_sims = genres_sims[(genres_sims['movie1_id'] != genres_sims['movie2_id']) & (genres_sims['similarity_score'] > 0.0)]

    
    
    def create_movie_crew_similarity_matrix(self):
        print("Creating movie crew similarity matrix...")
        #Combine the entries of the director and actors columns into a single list, then merge that list into a string
        crew = self.movies.apply(lambda row: row['director'] + row['actors'], axis=1)
        crew = crew.apply(lambda x: ' '.join(x))
                                                
        #Drop the director and actors columns from the movies dataset
        self.movies = self.movies.drop(columns = ['director', 'actors'])

        tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
        dtm = tfidf_vectorizer.fit_transform(crew)

        crew_sim_matrix = pd.DataFrame(cosine_similarity(dtm, dense_output = True))
        crew_sim_matrix.index = self.movies['tmdb_id'].values
        crew_sim_matrix.columns = self.movies['tmdb_id'].values

        #Reshape the similarity matrix to have columns (movie1_id, movie2_id, similarity_score)
        crew_sims = crew_sim_matrix.stack().reset_index()
        crew_sims.columns = ['movie1_id', 'movie2_id', 'similarity_score']

        #Drop rows where similarity score is 0 (no similarity) or 1 (self similarity)for storage efficiency
        self.crew_sims = crew_sims[(crew_sims['movie1_id'] != crew_sims['movie2_id']) & (crew_sims['similarity_score'] > 0.0)]


    def create_movie_content_similarity_matrix(self):
        print("Creating movie content similarity matrix...")
        #Create a content series by combining the keywords with the overview column, giving overview a higher weight
        content = self.movies.apply(lambda row: row['keywords'] + row['overview'] * 2, axis = 1)
        content = content.apply(lambda x: ' '.join(x))

        tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 0.01)
        dtm = tfidf_vectorizer.fit_transform(content)

        content_sim_matrix = pd.DataFrame(cosine_similarity(dtm, dense_output = True))
        content_sim_matrix.index = self.movies['tmdb_id'].values
        content_sim_matrix.columns = self.movies['tmdb_id'].values

        content_sims = content_sim_matrix.stack().reset_index()
        content_sims.columns = ['movie1_id', 'movie2_id', 'similarity_score']

        # Drop rows where similarity score is 0 (no similarity) or was calculated between the same rows for storage efficiency
        self.content_sims = content_sims[(content_sims['movie1_id'] != content_sims['movie2_id']) & (content_sims['similarity_score'] > 0.0)]

        

    def create_movie_rating_similarity_matrix(self):
        print("Creating movie rating similarity matrix...")
        correlation_matrix = self.movie_ratings.corr(method = 'pearson', min_periods = 7)
        correlation_matrix.index.name = 'movie1_id'
        correlation_matrix.columns.name = 'movie2_id'

        ratings_sims = correlation_matrix.stack().reset_index()
        ratings_sims.columns = ['movie1_id', 'movie2_id', 'similarity_score']

        #Drop rows where similarity score is 0 (no similarity) or similarity is calculated between the same movies for storage efficiency
        self.ratings_sims = ratings_sims[(ratings_sims['movie1_id'] != ratings_sims['movie2_id']) & (ratings_sims['similarity_score'] > 0.0)]

    def update_movies_dataset(self):
        print("Updating movies dataset...")
        #Don't need the overview and keywords columns after the similarity matrices are created
        self.movies = self.movies.drop(columns = ['overview', 'keywords'])
        self.movies = self.movies.drop_duplicates(subset = ['tmdb_id'])


    def replace_nan_with_none(self):
        print("Replacing NaN with '\\N'...")
        #Replace NaN values with '\\N' to be compatible with MySQL
        self.movies.fillna(value = '\\N', inplace = True)
        self.genres_sims.fillna(value = '\\N', inplace = True)
        self.crew_sims.fillna(value = '\\N', inplace = True)
        self.content_sims.fillna(value = '\\N', inplace = True)
        self.ratings_sims.fillna(value = '\\N', inplace = True)

    
    def enforce_foreign_keys(self):
        print("Enforcing foreign keys...")
        # movies_cleaned.csv contain the primary key (tmdb_id) for the other tables
        # The other tables' foreign keys (movie1_id and movie2_id) should only contain values of tmdb_id that appear in movies_cleaned.csv
        valid_ids = set(pd.to_numeric(self.movies['tmdb_id'], errors='coerce').dropna().astype(int).values)

        def _coerce_and_filter(df, col_names):
            if df is None or len(df) == 0:
                return df

            # Coerce id columns to numeric -> int, drop non-numeric
            for col in col_names:
                # Coerce id columns to numeric -> int
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].astype(int)
                # Filter rows to only valid ids present in movies
                df = df[df[col].isin(valid_ids)]

            # Drop rows where any of the id columns are NaN
            df = df.dropna(subset=[c for c in col_names])
                        
            return df

        # Apply to all similarity DataFrames
        self.genres_sims = _coerce_and_filter(self.genres_sims, ['movie1_id', 'movie2_id'])
        self.crew_sims = _coerce_and_filter(self.crew_sims, ['movie1_id', 'movie2_id'])
        self.content_sims = _coerce_and_filter(self.content_sims, ['movie1_id', 'movie2_id'])
        self.ratings_sims = _coerce_and_filter(self.ratings_sims, ['movie1_id', 'movie2_id'])

    def deduplicate_similarity_pairs(self):
        """Remove symmetric duplicates by canonicalizing pairs so movie1_id < movie2_id"""
        print("Deduplicating similarity pairs...")
        def _dedupe(df):
            # Canonical order: ensure movie1_id < movie2_id
            min_ids = df.loc[:, ['movie1_id', 'movie2_id']].min(axis=1) #goes row by row and chooses the smaller of movie1_id and movie2_id, returning a series
            max_ids = df.loc[:, ['movie1_id', 'movie2_id']].max(axis=1) #goes row by row and chooses the larger of movie1_id and movie2_id, returning a series
            
            #create a deep copy of the dataframe to do the deduplication operation on
            df = df.copy()
            df['movie1_id'] = min_ids
            df['movie2_id'] = max_ids
            #Now, the movie1_id column will contain the smaller of the two ids and the movie2_id column will contain the larger of the two ids
            
            # Keep one row per unordered pair, prefer highest similarity if available
            df = df.sort_values('similarity_score', ascending=False).drop_duplicates(subset=['movie1_id', 'movie2_id'], keep='first')
            return df

        # Apply to all similarity DataFrames
        self.genres_sims = _dedupe(self.genres_sims)
        self.crew_sims = _dedupe(self.crew_sims)
        self.content_sims = _dedupe(self.content_sims)
        self.ratings_sims = _dedupe(self.ratings_sims)

    def filter_similarity_matrix(self):
        """Filter the similarity matrices to only include rows where the similarity score is greater than or equal to the threshold (if the similarity score is below the threshold, the similarity is considered)"""
        def _filter(df, threshold):
            df = df[df['similarity_score'] >= threshold]
            return df
        print("Filtering similarity matrices...")
        self.genres_sims = _filter(self.genres_sims, 0.1)
        self.crew_sims = _filter(self.crew_sims, 0.1)
        self.content_sims = _filter(self.content_sims, 0.1)
        self.ratings_sims = _filter(self.ratings_sims, 0.1)


    def execute_preprocessing_pipeline(self):
        #Preprocessing pipeline order
        steps = [
            self.load_all_datasets,
            self.clean_movies,
            self.clean_keywords,
            self.clean_credits,
            self.clean_movie_ratings,
            self.merge_datasets,
            self.create_movie_genre_similarity_matrix,
            self.create_movie_crew_similarity_matrix,
            self.create_movie_content_similarity_matrix,
            self.create_movie_rating_similarity_matrix,
            self.update_movies_dataset,
            self.enforce_foreign_keys,
            self.deduplicate_similarity_pairs,
            self.filter_similarity_matrix,
            self.replace_nan_with_none
        ]

        # Iterate with progress bar
        for step in tqdm(steps, desc="Preprocessing Pipeline Progress"):
            step()  # execute the method


if __name__ == '__main__':
    preprocessor = MoviesPreprocessor()
    preprocessor.execute_preprocessing_pipeline()
    #print(preprocessor.movies.head())
    
    # Export cleaned data using relative paths
    preprocessor.movies.to_csv(preprocessor.output_path / 'movies_cleaned.csv', index=False)
    preprocessor.genres_sims.to_csv(preprocessor.output_path / 'genres_sims_cleaned.csv', index=False)
    preprocessor.crew_sims.to_csv(preprocessor.output_path / 'crew_sims_cleaned.csv', index=False)
    preprocessor.content_sims.to_csv(preprocessor.output_path / 'content_sims_cleaned.csv', index=False)
    preprocessor.ratings_sims.to_csv(preprocessor.output_path / 'ratings_sims_cleaned.csv', index=False)
    

       



