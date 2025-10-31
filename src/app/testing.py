import supabase
from fastapi import FastAPI
from dotenv import load_dotenv
import os 
import uvicorn


load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")

if __name__ == "__main__":
    supabase = supabase.create_client(supabase_url, supabase_key)   
    res = supabase.rpc("get_movie_recommendations", {
    "movie_id": 13
    }).execute()
    print(res)

