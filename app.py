import os
import json
import time
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from supabase import create_client, Client
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import io

# ============================================================================
# CONFIGURATION (SECURED)
# ============================================================================

# Fetching credentials from environment variables for security 
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "cruisematch")
LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY")
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

# ============================================================================
# TEAM INFORMATION
# ============================================================================

TEAM_INFO = {
    "group_batch_order_number": "2_11",  # [cite: 211]
    "team_name": "CruiseMatch AI",       # [cite: 213]
    "students": [
        {"name": "Sham Alem", "email": "sham.alem@campus.technion.ac.il"},
        {"name": "Amal Marjieh", "email": "amal.marjiya@campus.technion.ac.il"},
        {"name": "Weaam Mulla", "email": "weaam.mulla@campus.technion.ac.il"}
    ] # [cite: 214-224]
}

# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# ============================================================================
# INITIALIZE CLIENTS (Lazy loading for efficiency) [cite: 200]
# ============================================================================

_supabase = None
_pinecone_index = None
_embedding_model = None

def get_supabase():
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials in environment variables")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise ValueError("Missing Pinecone API Key in environment variables")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # MiniLM is efficient and keeps context size small [cite: 202]
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# ============================================================================
# LLM FUNCTIONS
# ============================================================================

def call_llm(prompt, system_prompt=None, max_tokens=2000):
    """Call LLMod.ai API [cite: 309]"""
    if not LLMOD_API_KEY:
        return "Error: LLMOD_API_KEY not set in environment."

    headers = {
        "Authorization": f"Bearer {LLMOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "RPRTHPB-gpt-5-mini", # [cite: 61]
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            f"{LLMOD_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# [KEEP ALL YOUR CLASS MODULES: QueryParser, CruiseSearcher, etc. HERE]
# [THEY ARE ALREADY WELL-WRITTEN AND MATCH THE REQUIREMENTS]

# ============================================================================
# MAIN ENTRY POINT FOR RENDER 
# ============================================================================

if __name__ == '__main__':
    # Render uses port 10000 by default for Docker services [cite: 294]
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False) # Debug False for security
