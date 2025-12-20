#!/usr/bin/env python3
import duckdb
import requests
import json
import re

# --- CONFIG ----------------------------------------------------------
DB_PATH = "kth_metadata.duckdb"
MODEL = "llama3.1:8b"            # adjust if you switch model
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MAX_CANDIDATES = 200
TOP_N = 20
BATCH_SIZE = 25                  # LLM scoring batch size
# ---------------------------------------------------------------------


def get_duckdb_connection():
    return duckdb.connect(DB_PATH)


def fetch_candidates(con):
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
    df = con.execute(query).fetch_df()
    return df


def score_batch_with_llm(batch):
    """
    One batch -> LLM. Returns list[dict] OR raises ValueError if output is not pure JSON array.
    """

    instruction = """
You are ranking KTH publications by how centrally they concern graphene.

For each item, assign:
- relevance: integer 0–3
  - 0 = not really about graphene (only incidental mention)
  - 1 = somewhat related
  - 2 = clearly about graphene but not central
  - 3 = graphene is a main or central focus
- short_reason: one short sentence explaining why.

You will receive a JSON array under the key "items".
Respond ONLY with a JSON array, in the same order as the input, in this shape:
[
  {"pid": 123, "relevance": 3, "short_reason": "..." },
  ...
]
Do NOT add any text before or after the JSON.
"""

    prompt_text = instruction + "\n\nitems:\n" + json.dumps(batch, ensure_ascii=False, indent=2)

    payload = {
        "model": MODEL,
        "prompt": prompt_text,
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    text = resp.json()["response"].strip()

    # Strict: must start with '[' and end with ']'
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("LLM did not return pure JSON array:\n" + text)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError("Could not parse JSON array from LLM response:\n" + stripped) from e

    return data


def score_with_llm_batched(items):
    all_scores = []
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        scores = score_batch_with_llm(batch)
        all_scores.extend(scores)
    return all_scores


# --- Fallback heuristic if LLM fails completely ----------------------


GRAPHENE_TERMS = [
    "graphene",
    "graphene oxide",
    "graphene nanocomposite",
    "2d material",
    "two-dimensional material",
]


def heuristic_relevance(rec):
    """
    rec: dict with title, abstract, keywords.
    Returns integer 0–3 based on simple term counting and locations.
    """
    title = (rec.get("title") or "").lower()
    abstract = (rec.get("abstract") or "").lower()
    keywords = (rec.get("keywords") or "").lower()

    text = " ".join([title, abstract, keywords])

    score = 0
    # term counts
    hits = sum(text.count(t) for t in GRAPHENE_TERMS)

    if hits == 0:
        return 0

    score += min(hits, 3)  # cap

    # bonus if in title
    if any(t in title for t in GRAPHENE_TERMS):
        score += 1

    # clamp 0–3
    if score > 3:
        score = 3
    if score < 0:
        score = 0
    return score


def main():
    con = get_duckdb_connection()

    print("Fetching candidate graphene-related publications from DuckDB...")
    df = fetch_candidates(con)
    print(f"Got {len(df)} candidates")

    if df.empty:
        print("No candidates found.")
        return

    candidates = df.to_dict(orient="records")

    llm_items = [
        {
            "pid": int(r["PID"]),
            "year": int(r["Year"]) if r["Year"] is not None else None,
            "title": r["Title"] or "",
            "abstract": (r["Abstract"] or "")[:2000],
            "keywords": r["Keywords"] or "",
        }
        for r in candidates
    ]

    use_llm = True
    scores = None

    print("Scoring candidates with LLM for graphene relevance...")
    try:
        scores = score_with_llm_batched(llm_items)
        print("LLM scoring succeeded.")
    except Exception as e:
        print("LLM scoring failed, falling back to heuristic scoring.")
        print("Reason:", repr(e))
        use_llm = False

    score_by_pid = {}
    if use_llm and scores:
        for s in scores:
            pid = int(s.get("pid"))
            score_by_pid[pid] = {
                "relevance": int(s.get("relevance", 0)),
                "short_reason": s.get("short_reason", ""),
            }

    enriched = []
    for r in candidates:
        pid = int(r["PID"])
        if use_llm and pid in score_by_pid:
            rel = score_by_pid[pid]["relevance"]
            reason = score_by_pid[pid]["short_reason"]
        else:
            # fallback heuristic
            rel = heuristic_relevance(
                {
                    "title": r["Title"],
                    "abstract": r["Abstract"],
                    "keywords": r["Keywords"],
                }
            )
            reason = "heuristic term-based relevance"

        enriched.append(
            {
                "relevance": rel,
                "pid": pid,
                "year": r["Year"],
                "title": r["Title"],
                "reason": reason,
            }
        )

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
