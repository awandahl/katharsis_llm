#!/usr/bin/env python3
import duckdb
import requests
import json

# --- CONFIGURE THESE -------------------------------------------------
DB_PATH = "kth_metadata.duckdb"   # path to your DuckDB file
MODEL = "llama3.2:1b"            # or whatever model name you're running in Ollama
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MAX_CANDIDATES = 200             # how many rows to pull from DuckDB before re-ranking
TOP_N = 20                       # how many top results to print
# ---------------------------------------------------------------------


def get_duckdb_connection():
    con = duckdb.connect(DB_PATH)
    return con


def fetch_candidates(con):
    # Static, robust graphene filter using LIKE (DuckDB LIKE is case-insensitive for ASCII).[web:246][web:254]
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

    df = con.execute(query).fetch_df()  # pandas DataFrame[web:253][web:259]
    return df


def score_with_llm(items):
    """
    items: list of dicts with pid, year, title, abstract, keywords.
    Returns list of dicts: {pid, relevance, short_reason}.
    """

    payload = {
        "model": MODEL,
        "prompt": json.dumps(
            {
                "instructions": """
You are ranking KTH publications by how centrally they concern graphene.

For each item, assign:
- relevance: integer 0–3
  - 0 = not really about graphene (only incidental mention)
  - 1 = somewhat related
  - 2 = clearly about graphene but not central
  - 3 = graphene is a main or central focus
- short_reason: one short sentence explaining why.

Respond ONLY as a JSON array (no extra text), one object per input item, in the same order:
[
  {"pid": ..., "relevance": ..., "short_reason": "..."},
  ...
]
""",
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    text = resp.json()["response"].strip()

    # Extract JSON array from any wrapping text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not parse JSON array from LLM response:\n" + text)

    data = json.loads(text[start : end + 1])
    return data


def main():
    con = get_duckdb_connection()

    print("Fetching candidate graphene-related publications from DuckDB...")
    df = fetch_candidates(con)
    print(f"Got {len(df)} candidates")

    if df.empty:
        print("No candidates found.")
        return

    candidates = df.to_dict(orient="records")

    # Prepare items to send to LLM
    llm_items = [
        {
            "pid": int(r["PID"]),
            "year": int(r["Year"]) if r["Year"] is not None else None,
            "title": r["Title"] or "",
            "abstract": (r["Abstract"] or "")[:2000],  # truncate if very long
            "keywords": r["Keywords"] or "",
        }
        for r in candidates
    ]

    print("Scoring candidates with LLM for graphene relevance...")
    scores = score_with_llm(llm_items)

    # Index scores by pid
    score_by_pid = {s["pid"]: s for s in scores}

    enriched = []
    for r in candidates:
        pid = int(r["PID"])
        s = score_by_pid.get(pid, {"relevance": 0, "short_reason": "no score"})
        enriched.append(
            {
                "relevance": int(s.get("relevance", 0)),
                "pid": pid,
                "year": r["Year"],
                "title": r["Title"],
                "reason": s.get("short_reason", ""),
            }
        )

    # Sort by relevance (desc), then by year (desc), then PID
    enriched.sort(key=lambda x: (x["relevance"], x["year"] or 0, x["pid"]), reverse=True)

    print(f"\nTop {min(TOP_N, len(enriched))} graphene-related hits:\n")
    for e in enriched[:TOP_N]:
        print(f"[{e['relevance']}] PID {e['pid']} ({e['year']}) {e['title']}")
        if e["reason"]:
            print(f"   -> {e['reason']}")
        print("")

    print("Done.")


if __name__ == "__main__":
    main()
