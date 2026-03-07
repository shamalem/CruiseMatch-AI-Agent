#!/usr/bin/env python3
"""
CruiseMatch AI - Autonomous Cruise Recommendation Agent
========================================================
Render-safe version:
- No local sentence-transformers loading
- No Torch model in web worker
- Uses Pinecone metadata filtering and text query fallback
"""

import os
import json
import time
import requests
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import io

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPABASE_URL = "https://qprqcxermzgogatflhgb.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = "cruisematch"

LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY")
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

MIN_CRUISE_NIGHTS = 3

SUPPORTED_REGIONS = {
    "Caribbean",
    "Mediterranean",
    "Alaska",
    "Northern Europe",
    "Asia Pacific",
    "South America",
    "Australia/New Zealand",
    "Transatlantic",
    "World Cruise",
    "Hawaii",
    "Bahamas",
    "Mexico",
}

# ============================================================================
# TEAM INFORMATION
# ============================================================================

TEAM_INFO = {
    "group_batch_order_number": "2_11",
    "team_name": "CruiseMatch AI",
    "students": [
        {"name": "Sham Alem", "email": "sham.alem@campus.technion.ac.il"},
        {"name": "Amal Marjieh", "email": "amal.marjiya@campus.technion.ac.il"},
        {"name": "Weaam Mulla", "email": "weaam.mulla@campus.technion.ac.il"},
    ],
}

# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ============================================================================
# HELPERS
# ============================================================================

def is_supported_region(region):
    if not region:
        return True
    return region.strip() in SUPPORTED_REGIONS


# ============================================================================
# CLIENTS (LAZY LOADING)
# ============================================================================

_supabase = None
_pinecone_index = None


def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        if not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_KEY env var is missing")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY env var is missing")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


# ============================================================================
# LLM FUNCTIONS
# ============================================================================

def call_llm(prompt, system_prompt=None, max_tokens=2000):
    if not LLMOD_API_KEY:
        return "Error calling LLM: LLMOD_API_KEY env var is missing"

    headers = {
        "Authorization": f"Bearer {LLMOD_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "RPRTHPB-gpt-5-mini",
        "messages": messages,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            f"{LLMOD_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


# ============================================================================
# AGENT MODULES
# ============================================================================

class QueryParser:
    @staticmethod
    def parse(user_query):
        system_prompt = """You are a query parser for a cruise recommendation system.
Extract the following parameters from the user's query and return them as JSON:
- budget_max: maximum budget in USD (integer or null)
- budget_min: minimum budget in USD (integer or null)
- duration_min: minimum nights (integer or null)
- duration_max: maximum nights (integer or null)
- region: cruise region like "Caribbean", "Mediterranean", "Alaska" etc (string or null)
- cabin_type: "Inside", "Ocean View", "Balcony", "Suite", or "Penthouse Suite" (string or null)
- cruise_line: specific cruise line if mentioned (string or null)
- family_friendly: true if they want family-friendly (boolean or null)
- adults_only: true if they want adults-only (boolean or null)
- experience_type: what kind of experience they want like "romantic", "adventure", "relaxation" (string or null)

Return ONLY valid JSON, no other text."""

        prompt = f'Parse this cruise search query: "{user_query}"'
        response = call_llm(prompt, system_prompt)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            parsed = json.loads(response.strip())

            if parsed.get("duration_min") is not None:
                parsed["duration_min"] = max(MIN_CRUISE_NIGHTS, int(parsed["duration_min"]))
            if parsed.get("duration_max") is not None:
                parsed["duration_max"] = max(MIN_CRUISE_NIGHTS, int(parsed["duration_max"]))

            return parsed
        except Exception:
            return {}


class CruiseSearcher:
    """
    Render-safe searcher:
    - no local embeddings
    - try Pinecone text query if supported
    - otherwise use Supabase filtering fallback
    """

    @staticmethod
    def _build_filters(filters):
        pinecone_filter = {}
        if filters:
            if filters.get("region"):
                pinecone_filter["region"] = {"$eq": filters["region"]}
            if filters.get("cabin_type"):
                pinecone_filter["cabin_type"] = {"$eq": filters["cabin_type"]}
            if filters.get("family_friendly") is True:
                pinecone_filter["family_friendly"] = {"$eq": True}
            if filters.get("adults_only") is True:
                pinecone_filter["adults_only"] = {"$eq": True}
            if filters.get("cruise_line"):
                pinecone_filter["cruise_line"] = {"$eq": filters["cruise_line"]}
        return pinecone_filter

    @staticmethod
    def _query_supabase(filters=None, limit=20):
        try:
            supabase = get_supabase()
            q = supabase.table("cruises").select(
                "cruise_id,cruise_line,ship_name,region,duration_nights,cabin_type,"
                "price_usd,rating,family_friendly,adults_only,all_inclusive,"
                "departure_port_country,departure_country"
            )

            if filters:
                if filters.get("region"):
                    q = q.eq("region", filters["region"])
                if filters.get("cabin_type"):
                    q = q.eq("cabin_type", filters["cabin_type"])
                if filters.get("family_friendly") is True:
                    q = q.eq("family_friendly", True)
                if filters.get("adults_only") is True:
                    q = q.eq("adults_only", True)
                if filters.get("cruise_line"):
                    q = q.eq("cruise_line", filters["cruise_line"])
                if filters.get("budget_max") is not None:
                    q = q.lte("price_usd", filters["budget_max"])
                if filters.get("budget_min") is not None:
                    q = q.gte("price_usd", filters["budget_min"])
                if filters.get("duration_max") is not None:
                    q = q.lte("duration_nights", filters["duration_max"])
                if filters.get("duration_min") is not None:
                    q = q.gte("duration_nights", filters["duration_min"])

            result = q.limit(limit).execute()
            return result.data or []

        except Exception as e:
            msg = str(e).lower()
            if "statement timeout" in msg or "57014" in msg:
                return []
            raise

    @staticmethod
    def search(query_text, filters=None, top_k=20):
        pinecone_filter = CruiseSearcher._build_filters(filters)

        try:
            index = get_pinecone_index()
            results = index.search(
                namespace="cruises",
                query={
                    "inputs": {"text": query_text},
                    "top_k": top_k,
                    "filter": pinecone_filter if pinecone_filter else None,
                },
                fields=["cruise_id", "region", "cabin_type", "cruise_line", "ship_name"]
            )

            matches = results.get("result", {}).get("hits", []) if isinstance(results, dict) else []
            if matches:
                cruise_ids = []
                score_map = {}

                for match in matches:
                    fields = match.get("fields", {})
                    cid = fields.get("cruise_id")
                    if cid:
                        cruise_ids.append(cid)
                        score_map[cid] = match.get("_score", match.get("score"))

                if cruise_ids:
                    supabase = get_supabase()
                    q = supabase.table("cruises").select(
                        "cruise_id,cruise_line,ship_name,region,duration_nights,cabin_type,"
                        "price_usd,rating,family_friendly,adults_only,all_inclusive,"
                        "departure_port_country,departure_country"
                    ).in_("cruise_id", cruise_ids)

                    if filters:
                        if filters.get("budget_max") is not None:
                            q = q.lte("price_usd", filters["budget_max"])
                        if filters.get("budget_min") is not None:
                            q = q.gte("price_usd", filters["budget_min"])
                        if filters.get("duration_max") is not None:
                            q = q.lte("duration_nights", filters["duration_max"])
                        if filters.get("duration_min") is not None:
                            q = q.gte("duration_nights", filters["duration_min"])

                    db_result = q.limit(top_k).execute()
                    db_rows = db_result.data or []

                    merged = []
                    for row in db_rows:
                        row["similarity_score"] = score_map.get(row.get("cruise_id"))
                        merged.append(row)

                    if merged:
                        return merged
        except Exception:
            pass

        return CruiseSearcher._query_supabase(filters=filters, limit=top_k)


class DestinationEvaluator:
    @staticmethod
    def evaluate(port_countries, experience_type):
        if not port_countries or not experience_type:
            return {}

        supabase = get_supabase()

        experience_map = {
            "romantic": "romance_score",
            "adventure": "adventure_score",
            "relaxation": "relaxation_score",
            "culture": "culture_score",
            "nature": "nature_score",
            "nightlife": "nightlife_score",
            "food": "food_scene_score",
            "beach": "beach_score",
            "family": "family_friendly_score",
        }

        score_column = experience_map.get(experience_type.lower(), "relaxation_score")

        result = (
            supabase.table("destinations")
            .select(f"country_name, city_name, {score_column}")
            .in_("country_name", port_countries)
            .order(score_column, desc=True)
            .limit(10)
            .execute()
        )

        return result.data


class ConstraintRelaxer:
    @staticmethod
    def _normalize(filters: dict) -> dict:
        f = dict(filters or {})
        for k, v in list(f.items()):
            if isinstance(v, str) and v.strip() == "":
                f[k] = None

        if f.get("duration_min") is not None:
            f["duration_min"] = max(MIN_CRUISE_NIGHTS, int(f["duration_min"]))
        if f.get("duration_max") is not None:
            f["duration_max"] = max(MIN_CRUISE_NIGHTS, int(f["duration_max"]))

        return f

    @staticmethod
    def relax(filters: dict, already_relaxed: set):
        f = ConstraintRelaxer._normalize(filters)

        if "duration_min_-1" not in already_relaxed and f.get("duration_min") is not None:
            old = int(f["duration_min"])
            f["duration_min"] = max(MIN_CRUISE_NIGHTS, old - 1)
            already_relaxed.add("duration_min_-1")
            return f, {
                "changed": "duration_min",
                "from": old,
                "to": f["duration_min"],
                "rule": "duration -1",
            }

        if "duration_max_+1" not in already_relaxed and f.get("duration_max") is not None:
            old = int(f["duration_max"])
            f["duration_max"] = max(MIN_CRUISE_NIGHTS, old + 1)
            already_relaxed.add("duration_max_+1")
            return f, {
                "changed": "duration_max",
                "from": old,
                "to": f["duration_max"],
                "rule": "duration +1",
            }

        if "duration_min_-2" not in already_relaxed and f.get("duration_min") is not None:
            old = int(f["duration_min"])
            f["duration_min"] = max(MIN_CRUISE_NIGHTS, old - 2)
            already_relaxed.add("duration_min_-2")
            return f, {
                "changed": "duration_min",
                "from": old,
                "to": f["duration_min"],
                "rule": "duration -2",
            }

        if "duration_max_+2" not in already_relaxed and f.get("duration_max") is not None:
            old = int(f["duration_max"])
            f["duration_max"] = max(MIN_CRUISE_NIGHTS, old + 2)
            already_relaxed.add("duration_max_+2")
            return f, {
                "changed": "duration_max",
                "from": old,
                "to": f["duration_max"],
                "rule": "duration +2",
            }

        if "cabin_any" not in already_relaxed and f.get("cabin_type") is not None:
            old = f["cabin_type"]
            f["cabin_type"] = None
            already_relaxed.add("cabin_any")
            return f, {
                "changed": "cabin_type",
                "from": old,
                "to": None,
                "rule": "remove cabin filter and allow any cabin type",
            }

        if "budget_+25" not in already_relaxed and f.get("budget_max") is not None:
            old = int(f["budget_max"])
            f["budget_max"] = int(old * 1.25)
            already_relaxed.add("budget_+25")
            return f, {
                "changed": "budget_max",
                "from": old,
                "to": f["budget_max"],
                "rule": "budget +25% (once)",
            }

        if "region_drop" not in already_relaxed and f.get("region") is not None:
            old = f["region"]
            f["region"] = None
            already_relaxed.add("region_drop")
            return f, {
                "changed": "region",
                "from": old,
                "to": None,
                "rule": "drop region (last resort)",
            }

        return f, {"changed": None}


class ResponseGenerator:
    @staticmethod
    def generate(user_query, cruises, destination_scores=None):
        system_prompt = """You are a friendly cruise travel advisor. Based on the search results provided,
create a helpful recommendation response. Be concise but informative.
Include the top 3-5 cruise recommendations with key details (cruise line, ship, region, duration, price).
Explain why each cruise might be a good fit based on the user's query."""

        cruise_summary = []
        for i, cruise in enumerate(cruises[:5], 1):
            if isinstance(cruise, dict):
                data = cruise.get("metadata", cruise)

                summary = f"{i}. {data.get('cruise_line', 'N/A')} - {data.get('ship_name', 'N/A')}\n"
                summary += f"   Region: {data.get('region', 'N/A')}\n"
                summary += f"   Duration: {data.get('duration_nights', 'N/A')} nights\n"
                summary += f"   Cabin: {data.get('cabin_type', 'N/A')}\n"
                summary += f"   Price: ${data.get('price_usd', 'N/A')}\n"
                summary += f"   Rating: {data.get('rating', 'N/A')}/5\n"

                features = []
                if data.get("family_friendly"):
                    features.append("family-friendly")
                if data.get("adults_only"):
                    features.append("adults-only")
                if data.get("all_inclusive"):
                    features.append("all-inclusive")
                if features:
                    summary += f"   Features: {', '.join(features)}\n"

                cruise_summary.append(summary)

        prompt = f"""User Query: "{user_query}"

Found Cruises:
{chr(10).join(cruise_summary)}

Please provide a helpful recommendation based on these options."""

        return call_llm(prompt, system_prompt, max_tokens=2000)


class CruiseMatchAgent:
    def __init__(self):
        self.query_parser = QueryParser()
        self.cruise_searcher = CruiseSearcher()
        self.destination_evaluator = DestinationEvaluator()
        self.constraint_relaxer = ConstraintRelaxer()
        self.response_generator = ResponseGenerator()

    def execute(self, user_prompt):
        steps = []
        start_time = time.time()

        try:
            step1_start = time.time()
            parsed_params = self.query_parser.parse(user_prompt)
            steps.append(
                {
                    "module": "QueryParser",
                    "prompt": {"user_query": user_prompt},
                    "response": parsed_params,
                    "duration_ms": int((time.time() - step1_start) * 1000),
                }
            )

            if parsed_params.get("region") and not is_supported_region(parsed_params.get("region")):
                total_time = int((time.time() - start_time) * 1000)
                return {
                    "status": "ok",
                    "error": None,
                    "response": f"No cruises found for the region '{parsed_params.get('region')}'. Try changing the region.",
                    "steps": steps + [
                        {
                            "module": "RegionValidator",
                            "prompt": {"region": parsed_params.get("region")},
                            "response": {
                                "supported": False,
                                "allowed_regions": sorted(list(SUPPORTED_REGIONS)),
                            },
                            "duration_ms": 0,
                        }
                    ],
                    "execution_time_ms": total_time,
                }

            step2_start = time.time()
            cruises = self.cruise_searcher.search(user_prompt, parsed_params)
            steps.append(
                {
                    "module": "CruiseSearcher",
                    "prompt": {"query": user_prompt, "filters": parsed_params},
                    "response": {"num_results": len(cruises), "top_cruises": cruises[:3]},
                    "duration_ms": int((time.time() - step2_start) * 1000),
                }
            )

            relax_attempts = 0
            current_filters = dict(parsed_params or {})
            already_relaxed = set()

            TARGET_MIN_RESULTS = 3
            MAX_RELAX_ATTEMPTS = 7

            while (not cruises or len(cruises) < TARGET_MIN_RESULTS) and relax_attempts < MAX_RELAX_ATTEMPTS:
                relax_attempts += 1
                r_start = time.time()

                new_filters, relax_info = self.constraint_relaxer.relax(current_filters, already_relaxed)

                steps.append(
                    {
                        "module": "ConstraintRelaxer",
                        "prompt": current_filters,
                        "response": {
                            "attempt": relax_attempts,
                            "relax_info": relax_info,
                            "new_filters": new_filters,
                        },
                        "duration_ms": int((time.time() - r_start) * 1000),
                    }
                )

                if relax_info.get("changed") is None:
                    break

                s_start = time.time()
                cruises = self.cruise_searcher.search(user_prompt, new_filters)
                steps.append(
                    {
                        "module": "CruiseSearcher(Relaxed)",
                        "prompt": {"query": user_prompt, "filters": new_filters},
                        "response": {"num_results": len(cruises), "top_cruises": cruises[:3]},
                        "duration_ms": int((time.time() - s_start) * 1000),
                    }
                )

                current_filters = new_filters

                if cruises and len(cruises) >= TARGET_MIN_RESULTS:
                    break

            if not cruises:
                total_time = int((time.time() - start_time) * 1000)
                return {
                    "status": "ok",
                    "error": None,
                    "response": "No cruises found even after relaxing constraints. Try changing budget, region, or duration wording.",
                    "steps": steps,
                    "execution_time_ms": total_time,
                }

            if parsed_params.get("experience_type") and cruises:
                step3_start = time.time()
                countries = list(
                    set(
                        [
                            c.get("departure_country")
                            or c.get("departure_port_country")
                            or c.get("metadata", {}).get("departure_country")
                            or c.get("metadata", {}).get("departure_port_country")
                            for c in cruises[:10]
                            if c
                        ]
                    )
                )
                countries = [c for c in countries if c]

                dest_scores = self.destination_evaluator.evaluate(
                    countries,
                    parsed_params.get("experience_type"),
                )
                steps.append(
                    {
                        "module": "DestinationEvaluator",
                        "prompt": {"countries": countries, "experience_type": parsed_params.get("experience_type")},
                        "response": dest_scores,
                        "duration_ms": int((time.time() - step3_start) * 1000),
                    }
                )

            step4_start = time.time()
            response_text = self.response_generator.generate(user_prompt, cruises)
            steps.append(
                {
                    "module": "ResponseGenerator",
                    "prompt": {"user_query": user_prompt, "num_cruises": len(cruises)},
                    "response": {"generated_text": (response_text[:200] + "...") if response_text else ""},
                    "duration_ms": int((time.time() - step4_start) * 1000),
                }
            )

            total_time = int((time.time() - start_time) * 1000)

            return {
                "status": "ok",
                "error": None,
                "response": response_text,
                "steps": steps,
                "execution_time_ms": total_time,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response": None,
                "steps": steps,
            }


agent = CruiseMatchAgent()

# ============================================================================
# FRONTEND
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CruiseMatch AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 10px; font-size: 2.5rem; }
        .subtitle { text-align: center; color: #8892b0; margin-bottom: 30px; }
        .input-section {
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }
        textarea {
            width: 100%;
            padding: 16px;
            border: 2px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            background: rgba(255,255,255,0.05);
            color: #fff;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
        }
        textarea:focus { outline: none; border-color: #4fc3f7; }
        button {
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
            color: #1a1a2e;
            border: none;
            padding: 14px 32px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            margin-top: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(79, 195, 247, 0.4); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .response-section {
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            display: none;
        }
        .response-section.visible { display: block; }
        .response-text {
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 20px;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        .steps-section { margin-top: 24px; }
        .step {
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid #4fc3f7;
        }
        .step-header { font-weight: 600; color: #4fc3f7; margin-bottom: 8px; }
        .step-content { font-size: 14px; color: #b0bec5; }
        pre { background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; }
        .loading { text-align: center; padding: 40px; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #4fc3f7;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }
        @keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }
        .examples { margin-top: 16px; }
        .example-btn {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            margin: 4px;
            cursor: pointer;
        }
        .example-btn:hover { background: rgba(255,255,255,0.2); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚢 CruiseMatch AI</h1>
        <p class="subtitle">Your Autonomous Cruise Recommendation Agent</p>

        <div class="input-section">
            <textarea id="prompt" placeholder="Describe your ideal cruise... e.g., 'Find me a romantic Caribbean cruise under $2000 for 7 nights'"></textarea>

            <div class="examples">
                <span style="color: #8892b0; font-size: 14px;">Try:</span>
                <button class="example-btn" onclick="setExample('Find me a romantic Caribbean cruise under $2000 for 7 nights')">Romantic Caribbean</button>
                <button class="example-btn" onclick="setExample('Family-friendly Alaska cruise with balcony cabin')">Family Alaska</button>
                <button class="example-btn" onclick="setExample('Adventure cruise to Mediterranean, 10-14 nights')">Mediterranean Adventure</button>
            </div>

            <button id="runBtn" onclick="runAgent()">🚀 Run Agent</button>
        </div>

        <div id="loading" class="response-section">
            <div class="loading">
                <div class="spinner"></div>
                <p>Finding the perfect cruise for you...</p>
            </div>
        </div>

        <div id="response" class="response-section">
            <h3 style="margin-bottom: 16px;">💬 Recommendation</h3>
            <div id="responseText" class="response-text"></div>

            <div class="steps-section">
                <h3 style="margin-bottom: 16px;">📋 Execution Steps</h3>
                <div id="steps"></div>
            </div>
        </div>
    </div>

    <script>
        function setExample(text) {
            document.getElementById('prompt').value = text;
        }

        async function runAgent() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }

            const btn = document.getElementById('runBtn');
            const loading = document.getElementById('loading');
            const response = document.getElementById('response');

            btn.disabled = true;
            loading.classList.add('visible');
            response.classList.remove('visible');

            try {
                const res = await fetch('/api/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });

                if (!res.ok) {
                    let msg = `HTTP ${res.status}`;
                    try {
                        const errData = await res.json();
                        msg = errData.error || errData.response || msg;
                    } catch (_) {
                        const text = await res.text();
                        if (text) msg = text;
                    }
                    throw new Error(msg);
                }

                const data = await res.json();

                document.getElementById('responseText').textContent =
                    data.response || data.error || 'No response';

                const stepsHtml = (data.steps || []).map(step => `
                    <div class="step">
                        <div class="step-header">${step.module}</div>
                        <div class="step-content">
                            <strong>Input:</strong>
                            <pre>${JSON.stringify(step.prompt, null, 2)}</pre>
                            <strong>Output:</strong>
                            <pre>${JSON.stringify(step.response, null, 2)}</pre>
                        </div>
                    </div>
                `).join('');

                document.getElementById('steps').innerHTML = stepsHtml;

                loading.classList.remove('visible');
                response.classList.add('visible');

            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                loading.classList.remove('visible');
            }
        }
    </script>
</body>
</html>
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/team_info", methods=["GET"])
def team_info():
    return jsonify(TEAM_INFO)


@app.route("/api/agent_info", methods=["GET"])
def agent_info():
    return jsonify(
        {
            "description": "CruiseMatch AI is an autonomous cruise recommendation agent that helps travelers find the perfect cruise based on their preferences, budget, and desired experiences.",
            "purpose": "To solve the problem of cruise selection by evaluating both the cruise itself (price, duration, cabin type) and the travel experience offered by departure/arrival locations (culture, nature, adventure).",
            "prompt_template": {
                "template": "Find me a {experience_type} cruise to {region} for {duration} nights under ${budget} with {cabin_type} cabin",
            },
            "prompt_examples": [
                {
                    "prompt": "Find me a romantic Caribbean cruise under $2000 for 7 nights",
                    "full_response": "Based on your preferences for a romantic Caribbean getaway under $2000 for 7 nights, I recommend: 1) Royal Caribbean's Symphony of the Seas - 7 nights, Balcony cabin at $1,850...",
                    "steps": [
                        {
                            "module": "QueryParser",
                            "prompt": {},
                            "response": {
                                "budget_max": 2000,
                                "region": "Caribbean",
                                "duration_max": 7,
                                "experience_type": "romantic",
                            },
                        },
                        {"module": "CruiseSearcher", "prompt": {}, "response": {"num_results": 15}},
                        {"module": "DestinationEvaluator", "prompt": {}, "response": {}},
                        {"module": "ResponseGenerator", "prompt": {}, "response": {}},
                    ],
                }
            ],
        }
    )


@app.route("/api/model_architecture", methods=["GET"])
def model_architecture():
    try:
        from PIL import Image, ImageDraw, ImageFont

        width, height = 900, 650
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
            font_title = font

        draw.text((280, 20), "CruiseMatch AI Architecture", fill="black", font=font_title)

        modules = [
            (80, 80, 340, 140, "User Query Input", "#E3F2FD"),
            (80, 160, 340, 220, "1. QueryParser\\n(LLM: Extract parameters)", "#BBDEFB"),
            (80, 240, 340, 300, "2. CruiseSearcher\\n(Pinecone/Supabase)", "#90CAF9"),
            (80, 320, 340, 380, "2b. ConstraintRelaxer\\n(Duration→Cabin→Budget→Region)", "#7FB3FF"),
            (80, 400, 340, 460, "3. DestinationEvaluator\\n(Supabase: Experience scores)", "#64B5F6"),
            (80, 480, 340, 540, "4. ResponseGenerator\\n(LLM: Create response)", "#42A5F5"),
            (80, 560, 340, 620, "Final Response + Steps", "#1E88E5"),
        ]

        datastores = [
            (540, 220, 820, 280, "Pinecone", "#C8E6C9"),
            (540, 310, 820, 370, "Supabase", "#A5D6A7"),
            (540, 400, 820, 460, "LLMod.ai", "#81C784"),
        ]

        for x1, y1, x2, y2, text, color in modules:
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=2)
            draw.text((x1 + 10, y1 + 10), text, fill="black", font=font)

        for x1, y1, x2, y2, text, color in datastores:
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=2)
            draw.text((x1 + 10, y1 + 10), text, fill="black", font=font)

        for i in range(len(modules) - 1):
            y = modules[i][3]
            draw.line([(210, y), (210, y + 20)], fill="black", width=2)
            draw.polygon([(205, y + 15), (215, y + 15), (210, y + 20)], fill="black")

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return send_file(img_bytes, mimetype="image/png")

    except ImportError:
        return jsonify({"error": "PIL not installed. Run: pip install Pillow"}), 500


@app.route("/api/execute", methods=["POST"])
def execute():
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "Missing 'prompt' in request body",
                        "response": None,
                        "steps": [],
                    }
                ),
                400,
            )

        user_prompt = data["prompt"]
        result = agent.execute(user_prompt)

        try:
            supabase = get_supabase()
            supabase.table("agent_logs").insert(
                {
                    "session_id": str(time.time()),
                    "user_prompt": user_prompt,
                    "agent_response": result.get("response"),
                    "steps": json.dumps(result.get("steps", [])),
                    "status": result.get("status"),
                    "error_message": result.get("error"),
                    "execution_time_ms": result.get("execution_time_ms"),
                }
            ).execute()
        except Exception:
            pass

        status_code = 200 if result.get("status") == "ok" else 500
        return jsonify(result), status_code

    except Exception as e:
        return (
            jsonify({"status": "error", "error": str(e), "response": None, "steps": []}),
            500,
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
