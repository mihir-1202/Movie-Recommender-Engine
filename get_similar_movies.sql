-- Creates the function to get similar movies (used by the API)

CREATE OR REPLACE FUNCTION get_movie_recommendations(
    movie_id BIGINT,
    w_genres NUMERIC DEFAULT 0.30,
    w_crew NUMERIC DEFAULT 0.20,
    w_content NUMERIC DEFAULT 0.40,
    w_ratings NUMERIC DEFAULT 0.10,
    limit_count INT DEFAULT 20
)
RETURNS TABLE (
    tmdb_id BIGINT,
    title TEXT,
    combined_score DOUBLE PRECISION
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH
    g AS (
        SELECT
            CASE WHEN gs.movie1_id = movie_id THEN gs.movie2_id ELSE gs.movie1_id END AS neighbor_id,
            COALESCE(gs.similarity_score, 0) * w_genres AS wscore
        FROM genres_sims gs
        WHERE gs.movie1_id = movie_id OR gs.movie2_id = movie_id
    ),
    c AS (
        SELECT
            CASE WHEN cs.movie1_id = movie_id THEN cs.movie2_id ELSE cs.movie1_id END AS neighbor_id,
            COALESCE(cs.similarity_score, 0) * w_crew AS wscore
        FROM crew_sims cs
        WHERE cs.movie1_id = movie_id OR cs.movie2_id = movie_id
    ),
    t AS (
        SELECT
            CASE WHEN ts.movie1_id = movie_id THEN ts.movie2_id ELSE ts.movie1_id END AS neighbor_id,
            COALESCE(ts.similarity_score, 0) * w_content AS wscore
        FROM content_sims ts
        WHERE ts.movie1_id = movie_id OR ts.movie2_id = movie_id
    ),
    r AS (
        SELECT
            CASE WHEN rs.movie1_id = movie_id THEN rs.movie2_id ELSE rs.movie1_id END AS neighbor_id,
            COALESCE(rs.similarity_score, 0) * w_ratings AS wscore
        FROM ratings_sims rs
        WHERE rs.movie1_id = movie_id OR rs.movie2_id = movie_id
    ),
    all_scores AS (
        SELECT neighbor_id, wscore FROM g
        UNION ALL SELECT neighbor_id, wscore FROM c
        UNION ALL SELECT neighbor_id, wscore FROM t
        UNION ALL SELECT neighbor_id, wscore FROM r
    ),
    agg AS (
        SELECT neighbor_id, SUM(wscore) AS combined_score
        FROM all_scores
        GROUP BY neighbor_id
    )
    SELECT
        a.neighbor_id AS tmdb_id,
        m.title,
        a.combined_score
    FROM agg a
    JOIN movies_metadata m ON m.tmdb_id = a.neighbor_id
    WHERE a.neighbor_id <> movie_id
    ORDER BY a.combined_score DESC
    LIMIT limit_count;
END;
$$;
