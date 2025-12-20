#!/usr/bin/env python3
import duckdb
import json
import requests

# --- CONFIG ----------------------------------------------------------
DB_PATH = "kth_metadata.duckdb"
MAX_CANDIDATES = 200       # how many rows to pull from DuckDB initially
TOP_N = 20                 # how many top items to send to the LLM
MODEL = "llama3.1:8b"      # your Ollama model name
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# ---------------------------------------------------------------------

GRAPHENE_TERMS = [
    "graphene",
    "graphene oxide",
    "graphene nanocomposite",
    "2d material",
    "two-dimensional material",
]


def heuristic_relevance(title, abstract, keywords):
    title_l = (title or "").lower()
    abstract_l = (abstract or "").lower()
    keywords_l = (keywords or "").lower()
    text = " ".join([title_l, abstract_l, keywords_l])

    hits = sum(text.count(t) for t in GRAPHENE_TERMS)
    if hits == 0:
        return 0

    score = min(hits, 3)
    if any(t in title_l for t in GRAPHENE_TERMS):
        score += 1
    return min(score, 3)


def fetch_candidates():
    con = duckdb.connect(DB_PATH)

    where_clause = """
    (
        Title    LIKE '%graphene%' OR
        Title    LIKE '%graphene oxide%' OR
        Title    LIKE '%graphene nanocomposite%' OR
        Title    LIKE '%2D material%' OR
        Abstract LIKE '%graphene%' OR
        Abstract LIKE '%graphene oxide%' OR
        Abstract LIKE '%graphene nanocomposite%' OR
        Abstract LIKE '%2D material%' OR
        Keywords LIKE '%graphene%' OR
        Keywords LIKE '%graphene oxide%' OR
        Keywords LIKE '%graphene nanocomposite%' OR
        Keywords LIKE '%2D material%'
    )
    """

    query = f"""
    SELECT
        PID,
        Year,
        Title,
        Abstract,
        Keywords
    FROM pub
    WHERE {where_clause}
    LIMIT {MAX_CANDIDATES}
    """

    df = con.execute(query).fetch_df()  # documented Python client use[web:253][web:259]
    con.close()
    return df


def build_top_items():
    df = fetch_candidates()
    print(f"Got {len(df)} candidates")

    records = df.to_dict(orient="records")

    enriched = []
    for r in records:
        rel = heuristic_relevance(r["Title"], r["Abstract"], r["Keywords"])
        enriched.append(
            {
                "relevance": rel,
                "pid": int(r["PID"]),
                "year": int(r["Year"]) if r["Year"] is not None else None,
                "title": r["Title"] or "",
                "abstract": (r["Abstract"] or "")[:800],   # truncate for prompt size
                "keywords": r["Keywords"] or "",
            }
        )

    # sort by relevance desc, then year desc, then pid
    enriched.sort(
        key=lambda x: (x["relevance"], x["year"] or 0, x["pid"]),
        reverse=True,
    )

    top_items = enriched[:TOP_N]
    print(f"Selected top {len(top_items)} items for LLM story")
    return top_items


def ask_llm_for_story(top_items):
    instruction = """
You will receive a list of KTH research publications about graphene.

Each item has:
- pid: internal id
- year: publication year
- title
- abstract (possibly truncated)
- keywords

Task:
1. Group these publications into 3–6 main research themes (for example: graphene devices, nanocomposites, energy storage, biosensing, 2D materials integration, etc.).
2. For each theme:
   - Give a short heading (a few words).
   - Write 3–5 sentences describing what this theme covers in the KTH graphene work.
3. Under each theme, list 3–5 representative publications as bullets, with:
   - PID and year,
   - title,
   - one short phrase (max ~20 words) about its specific contribution.

Be concise and structured. Use plain text or simple markdown-style headings and bullets.
Do NOT output JSON; write normal text only.
"""

    prompt = instruction + "\n\nPublications:\n" + json.dumps(
        top_items, ensure_ascii=False, indent=2
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    text = resp.json()["response"]  # standard Ollama generate API usage[web:249][web:261]
    return text


def main():
    top_items = build_top_items()

    print("\nTop items (heuristic ranking):\n")
    for it in top_items:
        print(f"[{it['relevance']}] PID {it['pid']} ({it['year']}) {it['title']}")

    print("\nAsking LLM to write a thematic overview...\n")
    story = ask_llm_for_story(top_items)
    print(story)


if __name__ == "__main__":
    main()
