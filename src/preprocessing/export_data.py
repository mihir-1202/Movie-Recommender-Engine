from data_cleaning import MoviesPreprocessor
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, insert
from dotenv import load_dotenv
import os
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

load_dotenv()

username = os.getenv("MYSQL_USERNAME")
password = os.getenv("MYSQL_PASSWORD")
host     = os.getenv("MYSQL_HOST")
port     = os.getenv("MYSQL_PORT")
database = os.getenv("MYSQL_DATABASE")


def connect_to_mysql():
    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}", echo = True)
    return engine

def define_tables(metadata):
    movies_metadata = Table('movies_metadata', metadata,
                            Column('tmdb_id', Integer, nullable = False, primary_key = True),
                            Column('title', String(255), nullable = False),
                            Column('imdb_score', Float, nullable = True)
                            )
    
    genres_sims = Table('genres_sims', metadata,
                                       Column('movie1_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                       Column('movie2_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                       Column('similarity_score', Float, nullable=True)
                                       )
    
    crew_sims = Table('crew_sims', metadata,
                                       Column('movie1_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                       Column('movie2_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                       Column('similarity_score', Float, nullable=True)
                                       )
    
    content_sims = Table('content_sims', metadata,
                                       Column('movie1_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                       Column('movie2_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                       Column('similarity_score', Float, nullable=True)
                                       )
    
    rating_sims = Table('ratings_sims', metadata,
                                        Column('movie1_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                        Column('movie2_id', Integer, ForeignKey('movies_metadata.tmdb_id')),
                                        Column('similarity_score', Float, nullable=True)
                                    )
    
    return movies_metadata, genres_sims, crew_sims, content_sims, rating_sims

"""
def populate_tables(tables, movies_preprocessor, dfs, engine):
    for table, df_name in zip(tables, dfs):
        with engine.connect() as conn:
            if not hasattr(movies_preprocessor, df_name):
                raise AttributeError(f"{df_name} attribute not found in MoviesPreprocessor")
            
            data = getattr(movies_preprocessor, df_name).to_dict('records')
            conn.execute(insert(table), data)"""


    
if __name__ == "__main__":
    try:
        #Connect to the MySQL database and define the necessary tables
        engine = connect_to_mysql()
        metadata = MetaData()
        tables = define_tables(metadata)

        #Drop all tables if they already exist to start fresh
        metadata.drop_all(engine)
        metadata.create_all(engine)
        print("Tables created successfully!")

        #Execute the preprocessing pipeline and populate the tables
        movie_preprocessor = MoviesPreprocessor()
        movie_preprocessor.execute_preprocessing_pipeline()
        
        df_names = ['movies', 'genres_sims', 'crew_sims', 'content_sims', 'ratings_sims']
        #populate_tables(tables, movie_preprocessor, df_names, engine)

    except Exception as e:
        print(f"Error creating tables: {e}")
    
    
   