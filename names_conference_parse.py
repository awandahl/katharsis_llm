#!/usr/bin/env python3
import duckdb
import pandas as pd
import requests
import json
import re

DB_PATH = "kth_metadata.duckdb"
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MAX_ROWS = 1000  # start small while testing


def connect():
    return duckdb.connect(DB_PATH)


def fetch_conferences(con, limit=MAX_ROWS):
    df = con.execute(f"""
        SELECT
            pid,
            name_seq,
            conference
        FROM names_conference
        WHERE conference IS NOT NULL
        LIMIT {limit}
    """).fetch_df()  # DuckDB → pandas[web:253][web:325]
    return df


def parse_with_llm(conf_string: str):
    """
    Ask LLM to classify the string into name/place/dates.
    Returns dict with conf_name, conf_place, conf_dates (may be empty).
    """
    if conf_string is None:
        return {"conf_name": "", "conf_place": "", "conf_dates": ""}

    instruction = """
You are cleaning conference metadata.

You will receive a single raw conference string like:
"38th Annual ACM Symposium on User Interface Software and Technology, UIST 2025, Busan, Korea, September 28 - October 1, 2025"

Extract:
- conf_name: the conference name and series (no location, no dates).
- conf_place: city/region + country (if present).
- conf_dates: date range or year, in a compact form (e.g. "28 Sep 2025 - 1 Oct 2025" or "July 24-28, 2004").

If something is missing, return an empty string for that field.

Respond ONLY as a single JSON object, for example:
{"conf_name": "...", "conf_place": "...", "conf_dates": "..." }
"""

    prompt = instruction + f"\n\nRaw conference string:\n{conf_string}\n\nJSON:"

    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
    )
    resp.raise_for_status()
    text = resp.json()["response"].strip()

    # Attempt to extract one JSON object from the response
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # fallback: treat whole string as name if parsing fails
        return {"conf_name": conf_string, "conf_place": "", "conf_dates": ""}

    json_str = text[start : end + 1]
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return {"conf_name": conf_string, "conf_place": "", "conf_dates": ""}

    # Normalize keys
    return {
        "conf_name": str(obj.get("conf_name", "") or ""),
        "conf_place": str(obj.get("conf_place", "") or ""),
        "conf_dates": str(obj.get("conf_dates", "") or ""),
    }


def main():
    con = connect()
    df = fetch_conferences(con)
    print(f"Fetched {len(df)} conference rows for parsing")

    rows = []
    for _, row in df.iterrows():
        raw = row["conference"]
        parsed = parse_with_llm(raw)
        rows.append(
            {
                "pid": int(row["pid"]),
                "name_seq": int(row["name_seq"]),
                "raw_conference": raw,
                "conf_name": parsed["conf_name"],
                "conf_place": parsed["conf_place"],
                "conf_dates": parsed["conf_dates"],
            }
        )

    out = pd.DataFrame(rows)
    print(out.head(20).to_string(index=False))

    # Write to a new DuckDB table for inspection
    con.execute("DROP TABLE IF EXISTS names_conference_parsed")
    con.execute("CREATE TABLE names_conference_parsed AS SELECT * FROM out")  # pandas → DuckDB[web:253][web:325]

    # Also keep a CSV snapshot
    out.to_csv("names_conference_parsed_sample.csv", index=False)

    con.close()
    print("Done. Wrote parsed data to names_conference_parsed and CSV.")


if __name__ == "__main__":
    main()
