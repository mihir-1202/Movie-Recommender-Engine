use movies;

-- Create the movies (parent table)
CREATE table movies_metadata
(
	tmdb_id INT UNSIGNED PRIMARY KEY,
    title TEXT,
    imdb_score float
);

-- Create the similarity score tables (children tables)
CREATE TABLE crew_sims
(
    movie1_id INT UNSIGNED,
    movie2_id INT UNSIGNED,
    similarity_score FLOAT,
    CONSTRAINT crew_fk1 FOREIGN KEY (movie1_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE,
    CONSTRAINT crew_fk2 FOREIGN KEY (movie2_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE
);

CREATE TABLE genres_sims
(
    movie1_id INT UNSIGNED,
    movie2_id INT UNSIGNED,
    similarity_score FLOAT,
    CONSTRAINT genres_fk1 FOREIGN KEY (movie1_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE,
    CONSTRAINT genres_fk2 FOREIGN KEY (movie2_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE
);

CREATE TABLE content_sims
(
    movie1_id INT UNSIGNED,
    movie2_id INT UNSIGNED,
    similarity_score FLOAT,
    CONSTRAINT content_fk1 FOREIGN KEY (movie1_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE,
    CONSTRAINT content_fk2 FOREIGN KEY (movie2_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE
);

CREATE TABLE ratings_sims
(
    movie1_id INT UNSIGNED,
    movie2_id INT UNSIGNED,
    similarity_score FLOAT,
    CONSTRAINT ratings_fk1 FOREIGN KEY (movie1_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE,
    CONSTRAINT ratings_fk2 FOREIGN KEY (movie2_id) REFERENCES movies_metadata(tmdb_id) ON DELETE CASCADE
);


-- Load all of the data into the tables
LOAD DATA INFILE 'C:/Coding Projects/Movie-Recommender-Engine/data_cleaned/movies_cleaned.csv'
INTO TABLE movies_metadata
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

LOAD DATA INFILE 'C:/Coding Projects/Movie-Recommender-Engine/data_cleaned/crew_sims_cleaned.csv'
INTO TABLE crew_sims
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

LOAD DATA INFILE 'C:/Coding Projects/Movie-Recommender-Engine/data_cleaned/genres_sims_cleaned.csv'
INTO TABLE genres_sims
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

LOAD DATA INFILE 'C:/Coding Projects/Movie-Recommender-Engine/data_cleaned/content_sims_cleaned.csv'
INTO TABLE genres_sims
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

LOAD DATA INFILE 'C:/Coding Projects/Movie-Recommender-Engine/data_cleaned/ratings_sims_cleaned.csv'
INTO TABLE genres_sims
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;


-- Delete all the rows from the tables
TRUNCATE crew_sims;
TRUNCATE genres_sims;
TRUNCATE ratings_sims;
TRUNCATE movies_metadata;


-- Drop the tables
DROP table content_sims;
DROP table crew_sims;
DROP table genres_sims;
DROP table ratings_sims;
DROP table movies_metadata;