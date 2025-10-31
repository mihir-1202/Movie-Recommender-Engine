import supabase
from fastapi import FastAPI
from dotenv import load_dotenv
import os 
import uvicorn

app = FastAPI()
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")
supabase = supabase.create_client(supabase_url, supabase_key)

@app.get("/")
def get_movie_recommendations(movie_id: int):
    res = supabase.rpc("get_movie_recommendations", {
    "movie_id": movie_id
    }).execute()
    
    movie_recommendations = [obj['title'] for obj in res.data]
    return movie_recommendations

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)