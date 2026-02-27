#!/usr/bin/env python3
"""
CruiseMatch AI - Autonomous Cruise Recommendation Agent (SECURE)
===============================================================

Endpoints:
- GET  /api/team_info
- GET  /api/agent_info
- GET  /api/model_architecture (PNG)
- POST /api/execute

Render:
Start Command:
  gunicorn app:app --bind 0.0.0.0:$PORT
Build Command:
  pip install -r requirements.txt
"""

import os
import json
import time
import requests
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from supabase import create_client
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import io

# ============================================================================
# CONFIGURATION (SECURED) - ENV VARS ONLY
# ============================================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "cruisematch")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "cruises")

LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY")
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
LLMOD_MODEL = os.environ.get("LLMOD_MODEL", "RPRTHPB-gpt-5-mini")

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
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# ============================================================================
# LAZY CLIENTS
# ============================================================================

_supabase = None
_pinecone_index = None
_embedding_model = None


def get_supabase():
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing SUPABASE_URL / SUPABASE_KEY env vars")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise ValueError("Missing PINECONE_API_KEY env var")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ============================================================================
# LLM HELPERS
# ============================================================================

def _extract_json(text: str):
    if not text:
        return None
    t = text.strip()

    if "```" in t:
        if "```json" in t:
            t = t.split("```json", 1)[1].split("```", 1)[0].strip()
        else:
            parts = t.split("```")
            if len(parts) >= 2:
                t = parts[1].strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(t[s:e+1])
        except Exception:
            return None
    return None


def call_llm(prompt: str, system_prompt: str | None = None, max_tokens: int = 1200):
    if not LLMOD_API_KEY:
        return {"ok": False, "error": "Missing LLMOD_API_KEY env var", "content": ""}

    headers = {"Authorization": f"Bearer {LLMOD_API_KEY}", "Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": LLMOD_MODEL, "messages": messages, "max_tokens": max_tokens}

    try:
        r = requests.post(f"{LLMOD_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return {"ok": True, "error": None, "content": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"ok": False, "error": str(e), "content": ""}


# ============================================================================
# AGENT MODULES
# ============================================================================

class QueryParser:
    @staticmethod
    def parse(user_query: str):
        system_prompt = (
            "You are a query parser for a cruise recommendation system.\n"
            "Extract parameters and return ONLY valid JSON with keys:\n"
            "budget_max, budget_min, duration_min, duration_max, region, cabin_type, cruise_line,\n"
            "family_friendly, adults_only, experience_type.\n"
            "Use null when missing."
        )
        prompt = f'Parse this cruise search query: "{user_query}"'
        llm = call_llm(prompt, system_prompt, max_tokens=700)
        if not llm["ok"]:
            return {"_error": llm["error"]}
        obj = _extract_json(llm["content"])
        return obj if isinstance(obj, dict) else {}


class CruiseSearcher:
    @staticmethod
    def search(query_text: str, filters: dict | None = None, top_k: int = 20):
        model = get_embedding_model()
        emb = model.encode(query_text).tolist()

        index = get_pinecone_index()

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

        res = index.query(
            vector=emb,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
            filter=pinecone_filter if pinecone_filter else None,
        )

        matches = []
        for m in getattr(res, "matches", []) or []:
            matches.append(
                {
                    "cruise_id": (m.metadata or {}).get("cruise_id"),
                    "similarity_score": getattr(m, "score", None),
                    "metadata": m.metadata or {},
                }
            )

        # Optional: numeric filtering via Supabase
        if not filters:
            return matches

        needs_db = any(
            [
                filters.get("budget_max"),
                filters.get("budget_min"),
                filters.get("duration_max"),
                filters.get("duration_min"),
            ]
        )
        if not needs_db:
            return matches

        cruise_ids = [x["cruise_id"] for x in matches if x.get("cruise_id")]
        if not cruise_ids:
            return matches

        supa = get_supabase()
        q = supa.table("cruises").select("*").in_("cruise_id", cruise_ids)

        if filters.get("budget_max"):
            q = q.lte("price_usd", filters["budget_max"])
        if filters.get("budget_min"):
            q = q.gte("price_usd", filters["budget_min"])
        if filters.get("duration_max"):
            q = q.lte("duration_nights", filters["duration_max"])
        if filters.get("duration_min"):
            q = q.gte("duration_nights", filters["duration_min"])

        out = q.execute().data or []
        by_id = {r.get("cruise_id"): r for r in out if r.get("cruise_id")}

        filtered = []
        for m in matches:
            cid = m.get("cruise_id")
            if cid in by_id:
                row = by_id[cid]
                row["similarity_score"] = m.get("similarity_score")
                filtered.append(row)

        return filtered


class DestinationEvaluator:
    @staticmethod
    def evaluate(countries: list[str], experience_type: str):
        if not countries or not experience_type:
            return []

        exp_map = {
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
        col = exp_map.get(experience_type.lower(), "relaxation_score")

        supa = get_supabase()
        res = (
            supa.table("destinations")
            .select(f"country_name, city_name, {col}")
            .in_("country_name", countries)
            .order(col, desc=True)
            .limit(10)
            .execute()
        )
        return res.data or []


class ResponseGenerator:
    @staticmethod
    def generate(user_query: str, cruises: list, destination_scores: list | None = None):
        system_prompt = (
            "You are a friendly cruise travel advisor.\n"
            "Return 3-5 recommendations with key details and short reasons.\n"
            "Be concise."
        )

        top = cruises[:5] if cruises else []
        summaries = []
        for i, c in enumerate(top, 1):
            data = c.get("metadata") if isinstance(c, dict) and "metadata" in c else c
            data = data or {}
            summaries.append(
                f"{i}. {data.get('cruise_line','N/A')} - {data.get('ship_name','N/A')}\n"
                f"   Region: {data.get('region','N/A')}\n"
                f"   Duration: {data.get('duration_nights','N/A')} nights\n"
                f"   Cabin: {data.get('cabin_type','N/A')}\n"
                f"   Price: ${data.get('price_usd','N/A')}\n"
                f"   Rating: {data.get('rating','N/A')}/5"
            )

        dest_txt = ""
        if destination_scores:
            dest_txt = "\n\nTop destinations for this experience:\n" + "\n".join(
                [f"- {d.get('city_name','?')}, {d.get('country_name','?')}" for d in destination_scores[:5]]
            )

        prompt = (
            f'User Query: "{user_query}"\n\n'
            f"Found Cruises:\n{chr(10).join(summaries) if summaries else 'No matches found.'}"
            f"{dest_txt}\n\n"
            "Write the final recommendation."
        )

        llm = call_llm(prompt, system_prompt, max_tokens=1200)
        if not llm["ok"]:
            return f"Error: {llm['error']}"
        return llm["content"]


class CruiseMatchAgent:
    def execute(self, user_prompt: str):
        steps = []
        t0 = time.time()

        def add_step(module, prompt_obj, response_obj, start_ts):
            steps.append(
                {
                    "module": module,
                    "prompt": prompt_obj,
                    "response": response_obj,
                    "duration_ms": int((time.time() - start_ts) * 1000),
                }
            )

        try:
            s1 = time.time()
            parsed = QueryParser.parse(user_prompt)
            add_step("QueryParser", {"user_query": user_prompt}, parsed, s1)

            s2 = time.time()
            cruises = CruiseSearcher.search(user_prompt, parsed if isinstance(parsed, dict) else {})
            add_step("CruiseSearcher", {"query": user_prompt, "filters": parsed}, {"num_results": len(cruises), "top_3": cruises[:3]}, s2)

            dest_scores = []
            exp_type = parsed.get("experience_type") if isinstance(parsed, dict) else None
            if exp_type and cruises:
                s3 = time.time()
                countries = []
                for c in cruises[:10]:
                    if not isinstance(c, dict):
                        continue
                    meta = c.get("metadata") if "metadata" in c else c
                    for key in ["departure_country", "arrival_country", "country_name"]:
                        v = (meta or {}).get(key)
                        if v:
                            countries.append(v)
                countries = sorted(list(set([x for x in countries if x])))
                dest_scores = DestinationEvaluator.evaluate(countries, exp_type)
                add_step("DestinationEvaluator", {"countries": countries, "experience_type": exp_type}, dest_scores, s3)

            s4 = time.time()
            final_text = ResponseGenerator.generate(user_prompt, cruises, dest_scores)
            add_step("ResponseGenerator", {"user_query": user_prompt, "num_cruises": len(cruises)}, {"generated_text_preview": (final_text[:250] + "...") if final_text else ""}, s4)

            return {
                "status": "ok",
                "error": None,
                "response": final_text,
                "steps": steps,
                "execution_time_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "response": None, "steps": steps}


agent = CruiseMatchAgent()

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.get("/api/team_info")
def api_team_info():
    return jsonify(TEAM_INFO)


@app.get("/api/agent_info")
def api_agent_info():
    return jsonify(
        {
            "description": "CruiseMatch AI is an autonomous cruise recommendation agent that matches users to cruises using semantic search + structured filtering.",
            "purpose": "Help users find the best cruise option based on budget, duration, region, cabin type, and experience preferences.",
            "prompt_template": {"template": "Find me a {experience_type} cruise to {region} for {duration} nights under ${budget} with a {cabin_type} cabin."},
            "prompt_examples": [
                {
                    "prompt": "Find me a romantic Caribbean cruise under $2000 for 7 nights",
                    "full_response": "Example output is generated live via /api/execute.",
                    "steps": [
                        {"module": "QueryParser", "prompt": {"user_query": "..."}, "response": {"budget_max": 2000, "region": "Caribbean"}},
                        {"module": "CruiseSearcher", "prompt": {"query": "...", "filters": {}}, "response": {"num_results": 20}},
                        {"module": "ResponseGenerator", "prompt": {"user_query": "...", "num_cruises": 20}, "response": {"generated_text_preview": "..."}},
                    ],
                }
            ],
        }
    )


@app.get("/api/model_architecture")
def api_model_architecture():
    try:
        from PIL import Image, ImageDraw, ImageFont

        w, h = 900, 620
        img = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
            title = font

        d.text((280, 20), "CruiseMatch AI Architecture", fill="black", font=title)

        modules = [
            (80, 80, 330, 135, "User Query", "#E3F2FD"),
            (80, 155, 330, 220, "1) QueryParser\n(LLM)", "#BBDEFB"),
            (80, 240, 330, 305, "2) CruiseSearcher\n(Pinecone + Supabase)", "#90CAF9"),
            (80, 325, 330, 390, "3) DestinationEvaluator\n(Supabase scores)", "#64B5F6"),
            (80, 410, 330, 475, "4) ResponseGenerator\n(LLM)", "#42A5F5"),
            (80, 495, 330, 560, "Response + Steps", "#1E88E5"),
        ]
        stores = [
            (520, 210, 820, 275, "Pinecone\nVector Index", "#C8E6C9"),
            (520, 300, 820, 365, "Supabase\nCruises + Destinations", "#A5D6A7"),
            (520, 390, 820, 455, "LLMod.ai\nLLM API", "#81C784"),
        ]

        for x1, y1, x2, y2, text, color in modules:
            d.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=2)
            d.text((x1 + 10, y1 + 8), text, fill="black", font=font)

        for x1, y1, x2, y2, text, color in stores:
            d.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=2)
            d.text((x1 + 10, y1 + 8), text, fill="black", font=font)

        for i in range(len(modules) - 1):
            y = modules[i][3]
            d.line([(205, y), (205, y + 20)], fill="black", width=2)
            d.polygon([(200, y + 16), (210, y + 16), (205, y + 20)], fill="black")

        d.line([(330, 270), (520, 242)], fill="green", width=2)
        d.line([(330, 270), (520, 332)], fill="green", width=2)
        d.line([(330, 185), (520, 422)], fill="blue", width=2)
        d.line([(330, 442), (520, 422)], fill="blue", width=2)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except ImportError:
        return jsonify({"error": "Pillow not installed. Add Pillow to requirements.txt"}), 500


@app.post("/api/execute")
def api_execute():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"status": "error", "error": "Missing 'prompt' in request body", "response": None, "steps": []}), 400
    return jsonify(agent.execute(prompt))


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CruiseMatch AI</title>
<style>
  body{font-family:Arial,Helvetica,sans-serif;background:#0b1220;color:#fff;margin:0;padding:20px}
  .wrap{max-width:900px;margin:0 auto}
  textarea{width:100%;min-height:110px;border-radius:12px;border:1px solid #2b3a5a;background:#121b2f;color:#fff;padding:14px;font-size:15px}
  button{margin-top:12px;padding:12px 18px;border:0;border-radius:12px;background:#4fc3f7;color:#001018;font-weight:700;cursor:pointer}
  button:disabled{opacity:.6;cursor:not-allowed}
  .card{margin-top:18px;background:#121b2f;border:1px solid #2b3a5a;border-radius:16px;padding:16px;display:none}
  .card.show{display:block}
  pre{background:#0b1220;border:1px solid #2b3a5a;border-radius:12px;padding:12px;overflow:auto;color:#cfe1ff}
  .step{margin-top:10px;border-left:4px solid #4fc3f7;padding-left:12px}
</style>
</head>
<body>
<div class="wrap">
  <h1>🚢 CruiseMatch AI</h1>
  <p style="color:#9fb0d0;margin-top:4px">Autonomous Cruise Recommendation Agent</p>
  <textarea id="prompt" placeholder="Describe your ideal cruise..."></textarea>
  <button id="runBtn" onclick="runAgent()">Run Agent</button>
  <div id="out" class="card">
    <h3>Recommendation</h3>
    <pre id="resp"></pre>
    <h3>Steps</h3>
    <div id="steps"></div>
  </div>
</div>

<script>
async function runAgent(){
  const btn = document.getElementById('runBtn');
  const out = document.getElementById('out');
  const resp = document.getElementById('resp');
  const stepsDiv = document.getElementById('steps');
  const prompt = document.getElementById('prompt').value.trim();
  if(!prompt){ alert('Please enter a prompt'); return; }

  btn.disabled = true;
  out.classList.remove('show');
  resp.textContent = 'Running...';
  stepsDiv.innerHTML = '';

  try{
    const r = await fetch('/api/execute', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt})
    });
    const data = await r.json();
    resp.textContent = data.response || data.error || 'No response';

    const steps = data.steps || [];
    stepsDiv.innerHTML = steps.map(s => `
      <div class="step">
        <b>${s.module}</b> (${s.duration_ms || 0} ms)
        <div><small>Prompt</small><pre>${JSON.stringify(s.prompt, null, 2)}</pre></div>
        <div><small>Response</small><pre>${JSON.stringify(s.response, null, 2)}</pre></div>
      </div>
    `).join('');

    out.classList.add('show');
  }catch(e){
    resp.textContent = 'Error: ' + e.message;
    out.classList.add('show');
  }finally{
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

# Local run (Render uses gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
