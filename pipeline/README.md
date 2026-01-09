
# \#!/usr/bin/env python3

import duckdb
import pandas as pd
import requests
import json
import re

# ---------------- CONFIG ----------------

DB_PATH = "kth_metadata.duckdb"

MODEL = "llama3:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MAX_ROWS = 200      \# start small to inspect, then increase/disable
SHOW_EVERY = 1      \# show streaming for every Nth row (1 = all)
MAX_SERIES_CANDIDATES = 5

# ---------------------------------------

def connect():
return duckdb.connect(DB_PATH)

# ---------- LLM streaming helper ----------

def stream_llm_json(prompt: str, show_stream: bool = True) -> str:
"""
Call Ollama with streaming. Optionally print the response as it arrives.
Return the final accumulated text.
"""
resp = requests.post(
OLLAMA_URL,
json={"model": MODEL, "prompt": prompt, "stream": True},
stream=True,
)
resp.raise_for_status()

    full_text = []
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))
        chunk = data.get("response", "")
        if show_stream and chunk:
            print(chunk, end="", flush=True)
        full_text.append(chunk)
        if data.get("done"):
            break
    
    if show_stream:
        print()  # newline after streaming
    return "".join(full_text)
    
# ---------- dblp series candidate retrieval ----------

def find_series_candidates(con, conf_name: str, max_candidates: int = MAX_SERIES_CANDIDATES):
"""
Heuristic: get a small candidate list of dblp_conference_series rows
for a given normalized conference name.

    Returns list of (series_slug, stream_iri, series_name).
    """
    
    if not conf_name:
        return []
    
    # crude acronym heuristic: last all-caps token of length >= 3
    tokens = re.findall(r"\b[A-Z]{3,}\b", conf_name)
    acronym = tokens[-1] if tokens else ""
    short = " ".join(conf_name.split()[:6])
    
    if acronym:
        query = """
            SELECT series_slug, stream_iri, series_name
            FROM dblp_conference_series
            WHERE lower(series_slug) = lower(?)
               OR series_name ILIKE ?
            LIMIT ?
        """
        params = [acronym.lower(), f"%{acronym}%", max_candidates]
    else:
        query = """
            SELECT series_slug, stream_iri, series_name
            FROM dblp_conference_series
            WHERE series_name ILIKE ?
            LIMIT ?
        """
        params = [f"%{short}%", max_candidates]
    
    rows = con.execute(query, params).fetchall()
    return rows
    def choose_series_with_llm(conf_name: str, conf_dates: str, candidates):
"""
Ask the LLM to choose the best matching conference series from candidates.

    candidates: list of (series_slug, stream_iri, series_name).
    Returns (series_slug, stream_iri, series_name, reason) or (None, None, None, "").
    """
    
    if not candidates:
        return (None, None, None, "")
    
    # Build a compact candidate list for the prompt
    cand_lines = []
    for i, (slug, iri, name) in enumerate(candidates, start=1):
        cand_lines.append(f"{i}. slug='{slug}', name='{name}'")
    cand_text = "\n".join(cand_lines)
    
    instruction = """
    You are matching a cleaned conference instance to its conference series in the dblp knowledge graph.

You get:

- conf_name: normalized conference instance name
- conf_dates: normalized conference dates (if any)
- A small list of candidate conference series, each with slug and name.

Task:

- Choose the single best matching conference series from the candidate list.
- Prefer exact or near-exact matches on name and acronym, ignoring year and local edition.
- If no candidate is clearly appropriate, return chosen_index = null.

Respond ONLY as JSON, for example:
{
"chosen_index": 1,        // 1-based index into the candidates list, or null
"reason": "short explanation, max 20 words"
}
"""

    prompt = (
        instruction
        + f"\n\nconf_name: {conf_name}\n"
          f"conf_dates: {conf_dates}\n\n"
          "Candidates:\n"
        + cand_text
        + "\n\nJSON:"
    )
    
    text = stream_llm_json(prompt, show_stream=False)
    
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return (None, None, None, "")
    
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return (None, None, None, "")
    
    idx = obj.get("chosen_index")
    reason = str(obj.get("reason", "") or "")
    if not isinstance(idx, int) or idx < 1 or idx > len(candidates):
        return (None, None, None, reason)
    
    slug, iri, name = candidates[idx - 1]
    return (slug, iri, name, reason)
    
# ---------- fetch conference rows ----------

def fetch_conferences(con, limit=MAX_ROWS):
df = con.execute(f"""
SELECT
pid,
name_seq,
conference
FROM names_conference
WHERE conference IS NOT NULL
USING SAMPLE {limit} ROWS  -- random sample of {limit} rows
""").fetch_df()
return df

# --- cheap pre-filter: does the string even look like it has a date? ---

HAS_YEAR = re.compile(r"\b(19|20)\d{2}\b")
HAS_MONTH = re.compile(
r"\b("
r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
r"January|February|March|April|June|July|August|September|October|November|December"
r")\b",
re.IGNORECASE,
)

def looks_like_has_date(text: str) -> bool:
if not text:
return False
t = str(text).strip()
return bool(HAS_YEAR.search(t) or HAS_MONTH.search(t))

# --- extra guard: skip obvious garbage / non‑conference text -----------

MIN_LEN_FOR_LLM = 10
MAX_LEN_FOR_LLM = 400  \# avoid huge blobs (abstracts, HTML, etc.)

HTML_TAG_RE = re.compile(r"<[^>]+>")
MANY_DIGITS_RE = re.compile(r"\d{6,}")  \# long numeric junk

def looks_like_conference_string(text: str) -> bool:
"""
Heuristic filter to avoid wasting LLM calls on garbage:
- Non-empty, within length window.
- Not dominated by HTML tags or extremely long digit runs.
"""
if text is None:
return False
t = str(text).strip()
if len(t) < MIN_LEN_FOR_LLM or len(t) > MAX_LEN_FOR_LLM:
return False
if HTML_TAG_RE.search(t):
return False
if MANY_DIGITS_RE.search(t):
return False
return True

# --- place normalization (post-LLM) ----------------------------------

def normalize_place(place: str) -> str:
"""
- Remove leading/trailing space.
- If the place is ALL CAPS (letters only), downcase then title-case.
- Otherwise leave as is.
"""
if not place:
return place
p = str(place).strip()
letters = "".join(ch for ch in p if ch.isalpha())
if letters and letters.isupper():
return p.lower().title()
return p

# --- conference name normalization ------------------------------------

SMALL_WORDS = {
"and", "or", "of", "on", "in", "for", "to", "the", "a", "an", "at", "by", "with",
}

ACRONYM_OVERRIDES = {
"eccomas": "ECCOMAS",
}

def normalize_conf_name(name: str) -> str:
"""
Normalize conference name to title-style capitalization:
- Preserve obvious acronyms (AIAA, IEEE, EMAS, IFAC, etc.) as full caps.
- Capitalize main words.
- Keep small words lowercase, except when first token or after a colon.
"""
if not name:
return name
text = str(name).strip()

    tokens = re.split(r"(\s+)", text)  # keep spaces
    result = []
    start_of_segment = True
    
    for tok in tokens:
        if tok.isspace():
            result.append(tok)
            continue
    
        word = tok
        trailing = ""
        m = re.match(r"^([A-Za-z0-9]+)(.*)$", tok)
        if m:
            word, trailing = m.group(1), m.group(2)
    
        lower_word = word.lower()
    
        if lower_word in ACRONYM_OVERRIDES:
            new_word = ACRONYM_OVERRIDES[lower_word]
        elif word.isupper() and len(word) > 1:
            new_word = word
        else:
            if start_of_segment:
                new_word = word[:1].upper() + word[1:].lower()
            else:
                if lower_word in SMALL_WORDS:
                    new_word = lower_word
                else:
                    new_word = word[:1].upper() + word[1:].lower()
    
        result.append(new_word + trailing)
        if ":" in tok:
            start_of_segment = True
        else:
            start_of_segment = False
    
    return "".join(result)
    
# --- parse normalized conf_dates into numeric fields ------------------

ISO_RANGE_RE = re.compile(
r"""
^
(?P<start>\d{4}(?:-\d{2}(?:-\d{2})?)?)      \# 2025 or 2025-07 or 2025-07-13
(?:\s*/\s*
(?P<end>\d{4}(?:-\d{2}(?:-\d{2})?)?)     \# same patterns
)?
\$
""",
re.VERBOSE,
)

def parse_iso_like_date(s: str):
"""Parse YYYY[-MM[-DD]] into (year, month, day) or (None, None, None)."""
if not s:
return (None, None, None)
parts = s.split("-")
try:
year = int(parts[0])
except ValueError:
return (None, None, None)
month = int(parts[1]) if len(parts) >= 2 else None
day = int(parts[2]) if len(parts) >= 3 else None
return (year, month, day)

def derive_dates_from_conf_dates(conf_dates: str):
"""
Expected formats for conf_dates (normalized by LLM):
- 'YYYY-MM-DD / YYYY-MM-DD'
- 'YYYY-MM-DD'
- 'YYYY-MM / YYYY-MM'
- 'YYYY / YYYY'
Returns (b_day, b_month, b_year, e_day, e_month, e_year).
"""
if not conf_dates:
return (None, None, None, None, None, None)

    m = ISO_RANGE_RE.match(conf_dates.strip())
    if not m:
        return (None, None, None, None, None, None)
    
    start_raw = m.group("start")
    end_raw = m.group("end") or start_raw
    
    sy, sm, sd = parse_iso_like_date(start_raw)
    ey, em, ed = parse_iso_like_date(end_raw)
    
    return (sd, sm, sy, ed, em, ey)
    
# --- conference order extraction (series number) -----------------------

ORDINAL_WORDS = {
"first": 1,
"second": 2,
"third": 3,
"fourth": 4,
"fifth": 5,
"sixth": 6,
"seventh": 7,
"eighth": 8,
"ninth": 9,
"tenth": 10,
"eleventh": 11,
"twelfth": 12,
"thirteenth": 13,
"fourteenth": 14,
"fifteenth": 15,
"sixteenth": 16,
"seventeenth": 17,
"eighteenth": 18,
"nineteenth": 19,
"twentieth": 20,
"twenty-first": 21,
"twenty-second": 22,
"twenty-third": 23,
"twenty-fourth": 24,
"twenty-fifth": 25,
"twenty-sixth": 26,
"twenty-seventh": 27,
"twenty-eighth": 28,
"twenty-ninth": 29,
"thirtieth": 30,
}

ORDINAL_NUMBER_RE = re.compile(r"\b(\d+)[’']?(st|nd|rd|th)\b", re.IGNORECASE)

def extract_conf_order(text: str):
"""
Extracts the conference order number (e.g. 5 for 'Fifth International...').
Returns int or None.
"""
if not text:
return None
t = str(text)

    m = ORDINAL_NUMBER_RE.search(t)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    
    words = re.findall(r"[A-Za-z\-]+", t)
    for w in words:
        key = w.lower()
        if key in ORDINAL_WORDS:
            return ORDINAL_WORDS[key]
    
    return None
    
# --- LLM parsing with simple in-memory cache --------------------------

_llm_cache = {}

def parse_with_llm(conf_string: str, show_stream: bool = True):
"""
Ask LLM to classify the string into name/place/dates.
Returns dict with:
conf_name, conf_place, conf_dates (normalized text), note.
"""

    if conf_string is None:
        return {
            "conf_name": "",
            "conf_place": "",
            "conf_dates": "",
            "note": "",
        }
    
    if conf_string in _llm_cache:
        cached = _llm_cache[conf_string].copy()
        return cached
    
    instruction = """
    You are cleaning conference metadata.

You will receive a single raw conference string like:
"38th Annual ACM Symposium on User Interface Software and Technology, UIST 2025, Busan, Korea, September 28 - October 1, 2025"

Important notes about conference names and years:

- Some conference names legitimately include a year or acronym+year as part of the name, for example:
    - "ATTCE 2001-Automotive and Transport Technology Congress and Exhibition"
    - "European Congress on Computational Methods in Applied Sciences and Engineering, ECCOMAS 2004"
    - "AMIF 2002, Applied Mathematics for Industrial Flow Problems, Third International Conference"
    - "International Conference on Fatigue Crack Path (FCP 2003)"
- In these cases, the year and acronym belong to the conference NAME and must stay inside conf_name.
- If there is no other explicit date information (no clear date range), this year is a good candidate for both the begin and end year.

For conf_name:

- Preserve acronyms in uppercase EXACTLY as they appear in the raw string (e.g. AIAA, IEEE, IFAC, EMAS, ATTCE).
- Normalize capitalization in a title-like style for the rest of the words:
    - Capitalize main words.
    - Keep small connector words (and, of, on, in, for, to, the, a, an, at, by, with) lowercase,
except when they are the first word or follow a colon.

For conf_place:

- Normalize capitalization (no ALL-CAPS; use "Strasbourg, France" not "STRASBOURG, FRANCE").
- When it is obvious, use the correct local spelling with diacritics for city names.

For dates, ALWAYS normalize conf_dates into an ISO-like format:

- If a full range is available:
    - conf_dates: "YYYY-MM-DD / YYYY-MM-DD"
- If only a single day is known:
    - conf_dates: "YYYY-MM-DD"
- If only month and year are known (no specific days):
    - conf_dates: "YYYY-MM / YYYY-MM"
- If a year appears only as part of the conference name or acronym, and there is no explicit month in the date part:
    - conf_dates: "YYYY / YYYY"
- If only a year is known:
    - conf_dates: "YYYY / YYYY"
Do NOT invent specific months or days when only a year is mentioned.

Extract:

- conf_name
- conf_place
- conf_dates
- note: a very short explanation of how you interpreted the string (max 20 words).

Respond ONLY as a single JSON object:
{
"conf_name": "...",
"conf_place": "...",
"conf_dates": "2004-04-27 / 2004-04-29",
"note": "..."
}
"""

    prompt = instruction + f"\n\nRaw conference string:\n{conf_string}\n\nJSON:"
    
    text = stream_llm_json(prompt, show_stream=show_stream)
    
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        result = {
            "conf_name": conf_string,
            "conf_place": "",
            "conf_dates": "",
            "note": "fallback: could not parse JSON",
        }
        _llm_cache[conf_string] = result
        return result
    
    json_str = text[start : end + 1]
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        result = {
            "conf_name": conf_string,
            "conf_place": "",
            "conf_dates": "",
            "note": "fallback: JSON decode error",
        }
        _llm_cache[conf_string] = result
        return result
    
    result = {
        "conf_name": normalize_conf_name(str(obj.get("conf_name", "") or "")),
        "conf_place": normalize_place(obj.get("conf_place", "") or ""),
        "conf_dates": str(obj.get("conf_dates", "") or ""),
        "note": str(obj.get("note", "") or ""),
    }
    _llm_cache[conf_string] = result
    return result
    
# ---------- main pipeline ----------

def main():
con = connect()
df = fetch_conferences(con)
total = len(df)
print(f"Fetched {total} conference rows for parsing")

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        raw = row["conference"]
        pid = int(row["pid"])
        name_seq = int(row["name_seq"])
    
        show_stream = (i % SHOW_EVERY == 0)
    
        print(f"\n=== {i}/{total} PID {pid} name_seq {name_seq} ===")
        print("RAW:", raw)
    
        if looks_like_conference_string(raw) and looks_like_has_date(raw):
            if show_stream:
                print("LLM output (streaming):")
            parsed = parse_with_llm(raw, show_stream=show_stream)
        else:
            parsed = {
                "conf_name": normalize_conf_name(raw),
                "conf_place": "",
                "conf_dates": "",
                "note": "no date detected or skipped by heuristic",
            }
    
        b_day, b_month, b_year, e_day, e_month, e_year = derive_dates_from_conf_dates(
            parsed["conf_dates"]
        )
        conf_order = extract_conf_order(parsed["conf_name"])
    
        # dblp candidates + LLM choice
        candidates = find_series_candidates(con, parsed["conf_name"])
        series_slug, stream_iri, series_name, series_reason = choose_series_with_llm(
            parsed["conf_name"], parsed["conf_dates"], candidates
        )
    
        print(
            "PARSED:",
            f"name='{parsed['conf_name']}' | place='{parsed['conf_place']}' | "
            f"dates='{parsed['conf_dates']}' | order={conf_order}",
        )
        if parsed.get("note"):
            print("NOTE:", parsed["note"])
    
        if series_slug or stream_iri or series_name:
            print(
                "DBLP:",
                f"slug='{series_slug}' | iri='{stream_iri}' | series='{series_name}'",
            )
            if series_reason:
                print("DBLP_REASON:", series_reason)
        else:
            print("DBLP: no series match")
    
        # extra spacing between entries
        print()
        print()
    
        rows.append(
            {
                "pid": pid,
                "name_seq": name_seq,
                "raw_conference": raw,
                "conf_name": parsed["conf_name"],
                "conf_place": parsed["conf_place"],
                "conf_dates": parsed["conf_dates"],
                "conf_begin_date_day": b_day,
                "conf_begin_date_month": b_month,
                "conf_begin_date_year": b_year,
                "conf_end_date_day": e_day,
                "conf_end_date_month": e_month,
                "conf_end_date_year": e_year,
                "conf_order": conf_order,
                "conf_series_slug": series_slug,
                "conf_series_stream_iri": stream_iri,
                "conf_series_name": series_name,
                "conf_series_match_reason": series_reason,
                "note": parsed["note"],
            }
        )
    
    out = pd.DataFrame(rows)
    print("\nSample of parsed output:")
    print(out.head(20).to_string(index=False))
    
    con.execute("DROP TABLE IF EXISTS names_conference_parsed")
    con.sql("CREATE TABLE names_conference_parsed AS SELECT * FROM out")
    
    out.to_csv("names_conference_parsed_sample.csv", index=False)
    
    con.close()
    print("\nDone. Wrote parsed data to 'names_conference_parsed' and CSV.")
    if __name__ == "__main__":
main()

Yes, this script can be cleanly split into separate modules and orchestrated from a thin main file. The logic is already nicely factored into pieces that map almost 1:1 to separate files.

Below is a concrete refactor plan that matches what you described, while preserving current behaviour.

## Target layout

```text
confmeta/
  __init__.py

  config.py          # paths, model, limits
  db_io.py          # connect, fetch_conferences, write_parsed_table

  regex_utils.py    # looks_like_*, normalize_*, date parsing, order extraction
  llm_parse.py      # conference parsing via LLM (name/place/date)
  llm_series.py     # series candidate selection + LLM re-ranking (dblp etc.)

  pipeline.py       # orchestrator: glue everything together
```

You can keep this as a simple package (no install needed; just run with `PYTHONPATH` or from repo root).

Below are minimal versions of each file, adapted directly from your current script.

## config.py

```python
DB_PATH = "kth_metadata.duckdb"

MODEL = "llama3:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MAX_ROWS = 200
SHOW_EVERY = 1
MAX_SERIES_CANDIDATES = 5
```


## db_io.py

```python
import duckdb
import pandas as pd
from .config import DB_PATH

def connect():
    return duckdb.connect(DB_PATH)

def fetch_conferences(con, limit):
    return con.execute(f"""
        SELECT
            pid,
            name_seq,
            conference
        FROM names_conference
        WHERE conference IS NOT NULL
        USING SAMPLE {limit} ROWS
    """).fetch_df()

def write_parsed_table(con, df, table_name="names_conference_parsed"):
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    # DuckDB replacement scan on pandas df
    con.sql(f"CREATE TABLE {table_name} AS SELECT * FROM df")
```


## regex_utils.py (all regexp / normalization logic, no LLM)

```python
import re

MIN_LEN_FOR_LLM = 10
MAX_LEN_FOR_LLM = 400

HAS_YEAR = re.compile(r"\b(19|20)\d{2}\b")
HAS_MONTH = re.compile(
    r"\b("
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|October|November|December"
    r")\b",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MANY_DIGITS_RE = re.compile(r"\d{6,}")

SMALL_WORDS = {
    "and", "or", "of", "on", "in", "for", "to", "the", "a", "an", "at", "by", "with",
}
ACRONYM_OVERRIDES = {"eccomas": "ECCOMAS"}

ISO_RANGE_RE = re.compile(
    r"""
    ^
    (?P<start>\d{4}(?:-\d{2}(?:-\d{2})?)?)
    (?:\s*/\s*
       (?P<end>\d{4}(?:-\d{2}(?:-\d{2})?)?)
    )?
    $
    """,
    re.VERBOSE,
)

ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twenty-second": 22,
    "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25,
    "twenty-sixth": 26, "twenty-seventh": 27, "twenty-eighth": 28,
    "twenty-ninth": 29, "thirtieth": 30,
}
ORDINAL_NUMBER_RE = re.compile(r"\b(\d+)[’']?(st|nd|rd|th)\b", re.IGNORECASE)


def looks_like_has_date(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return bool(HAS_YEAR.search(t) or HAS_MONTH.search(t))


def looks_like_conference_string(text: str) -> bool:
    if text is None:
        return False
    t = str(text).strip()
    if len(t) < MIN_LEN_FOR_LLM or len(t) > MAX_LEN_FOR_LLM:
        return False
    if HTML_TAG_RE.search(t):
        return False
    if MANY_DIGITS_RE.search(t):
        return False
    return True


def normalize_place(place: str) -> str:
    if not place:
        return place
    p = str(place).strip()
    letters = "".join(ch for ch in p if ch.isalpha())
    if letters and letters.isupper():
        return p.lower().title()
    return p


def normalize_conf_name(name: str) -> str:
    if not name:
        return name
    text = str(name).strip()

    tokens = re.split(r"(\s+)", text)
    result = []
    start_of_segment = True

    for tok in tokens:
        if tok.isspace():
            result.append(tok)
            continue

        word = tok
        trailing = ""
        m = re.match(r"^([A-Za-z0-9]+)(.*)$", tok)
        if m:
            word, trailing = m.group(1), m.group(2)

        lower_word = word.lower()

        if lower_word in ACRONYM_OVERRIDES:
            new_word = ACRONYM_OVERRIDES[lower_word]
        elif word.isupper() and len(word) > 1:
            new_word = word
        else:
            if start_of_segment:
                new_word = word[:1].upper() + word[1:].lower()
            else:
                if lower_word in SMALL_WORDS:
                    new_word = lower_word
                else:
                    new_word = word[:1].upper() + word[1:].lower()

        result.append(new_word + trailing)
        start_of_segment = ":" in tok

    return "".join(result)


def parse_iso_like_date(s: str):
    if not s:
        return (None, None, None)
    parts = s.split("-")
    try:
        year = int(parts[0])
    except ValueError:
        return (None, None, None)
    month = int(parts[1]) if len(parts) >= 2 else None
    day = int(parts[2]) if len(parts) >= 3 else None
    return (year, month, day)


def derive_dates_from_conf_dates(conf_dates: str):
    if not conf_dates:
        return (None, None, None, None, None, None)

    m = ISO_RANGE_RE.match(conf_dates.strip())
    if not m:
        return (None, None, None, None, None, None)

    start_raw = m.group("start")
    end_raw = m.group("end") or start_raw

    sy, sm, sd = parse_iso_like_date(start_raw)
    ey, em, ed = parse_iso_like_date(end_raw)

    return (sd, sm, sy, ed, em, ey)


def extract_conf_order(text: str):
    if not text:
        return None
    t = str(text)

    m = ORDINAL_NUMBER_RE.search(t)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    words = re.findall(r"[A-Za-z\-]+", t)
    for w in words:
        key = w.lower()
        if key in ORDINAL_WORDS:
            return ORDINAL_WORDS[key]

    return None
```


## llm_parse.py (conference name/place/date via LLM)

```python
import json
import requests
from .config import MODEL, OLLAMA_URL
from .regex_utils import normalize_conf_name, normalize_place

_llm_cache = {}


def stream_llm_json(prompt: str, show_stream: bool = True) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": True},
        stream=True,
    )
    resp.raise_for_status()

    full_text = []
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))
        chunk = data.get("response", "")
        if show_stream and chunk:
            print(chunk, end="", flush=True)
        full_text.append(chunk)
        if data.get("done"):
            break

    if show_stream:
        print()
        print()
    return "".join(full_text)


def parse_with_llm(conf_string: str, show_stream: bool = True):
    if conf_string is None:
        return {
            "conf_name": "",
            "conf_place": "",
            "conf_dates": "",
            "note": "",
        }

    if conf_string in _llm_cache:
        return _llm_cache[conf_string].copy()

    instruction = """<your existing long instruction block unchanged>"""

    prompt = instruction + f"\n\nRaw conference string:\n{conf_string}\n\nJSON:"

    text = stream_llm_json(prompt, show_stream=show_stream)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        result = {
            "conf_name": conf_string,
            "conf_place": "",
            "conf_dates": "",
            "note": "fallback: could not parse JSON",
        }
        _llm_cache[conf_string] = result
        return result

    json_str = text[start : end + 1]
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        result = {
            "conf_name": conf_string,
            "conf_place": "",
            "conf_dates": "",
            "note": "fallback: JSON decode error",
        }
        _llm_cache[conf_string] = result
        return result

    result = {
        "conf_name": normalize_conf_name(str(obj.get("conf_name", "") or "")),
        "conf_place": normalize_place(obj.get("conf_place", "") or ""),
        "conf_dates": str(obj.get("conf_dates", "") or ""),
        "note": str(obj.get("note", "") or ""),
    }
    _llm_cache[conf_string] = result
    return result
```


## llm_series.py (dblp lookup + LLM re‑ranking)

```python
import json
import re
from .config import MAX_SERIES_CANDIDATES
from .llm_parse import stream_llm_json


def find_series_candidates(con, conf_name: str, max_candidates: int = MAX_SERIES_CANDIDATES):
    if not conf_name:
        return []

    tokens = re.findall(r"\b[A-Z]{3,}\b", conf_name)
    acronym = tokens[-1] if tokens else ""
    short = " ".join(conf_name.split()[:6])

    if acronym:
        query = """
            SELECT series_slug, stream_iri, series_name
            FROM dblp_conference_series
            WHERE lower(series_slug) = lower(?)
               OR series_name ILIKE ?
            LIMIT ?
        """
        params = [acronym.lower(), f"%{acronym}%", max_candidates]
    else:
        query = """
            SELECT series_slug, stream_iri, series_name
            FROM dblp_conference_series
            WHERE series_name ILIKE ?
            LIMIT ?
        """
        params = [f"%{short}%", max_candidates]

    return con.execute(query, params).fetchall()


def choose_series_with_llm(conf_name: str, conf_dates: str, candidates):
    if not candidates:
        return (None, None, None, "")

    cand_lines = []
    for i, (slug, iri, name) in enumerate(candidates, start=1):
        cand_lines.append(f"{i}. slug='{slug}', name='{name}'")
    cand_text = "\n".join(cand_lines)

    instruction = """
You are matching a cleaned conference instance to its conference series in the dblp knowledge graph.

You get:
- conf_name: normalized conference instance name
- conf_dates: normalized conference dates (if any)
- A small list of candidate conference series, each with slug and name.

Task:
- Choose the single best matching conference series from the candidate list.
- Prefer exact or near-exact matches on name and acronym, ignoring year and local edition.
- If no candidate is clearly appropriate, return chosen_index = null.

Respond ONLY as JSON, for example:
{
  "chosen_index": 1,
  "reason": "short explanation, max 20 words"
}
"""

    prompt = (
        instruction
        + f"\n\nconf_name: {conf_name}\n"
          f"conf_dates: {conf_dates}\n\n"
          "Candidates:\n"
        + cand_text
        + "\n\nJSON:"
    )

    text = stream_llm_json(prompt, show_stream=False)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return (None, None, None, "")

    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return (None, None, None, "")

    idx = obj.get("chosen_index")
    reason = str(obj.get("reason", "") or "")
    if not isinstance(idx, int) or idx < 1 or idx > len(candidates):
        return (None, None, None, reason)

    slug, iri, name = candidates[idx - 1]
    return (slug, iri, name, reason)
```


## pipeline.py (orchestrator)

```python
#!/usr/bin/env python3
import pandas as pd
from .config import MAX_ROWS, SHOW_EVERY
from .db_io import connect, fetch_conferences, write_parsed_table
from .regex_utils import (
    looks_like_conference_string,
    looks_like_has_date,
    derive_dates_from_conf_dates,
    extract_conf_order,
    normalize_conf_name,
)
from .llm_parse import parse_with_llm
from .llm_series import find_series_candidates, choose_series_with_llm


def main():
    con = connect()
    df = fetch_conferences(con, MAX_ROWS)
    total = len(df)
    print(f"Fetched {total} conference rows for parsing")

    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        raw = row["conference"]
        pid = int(row["pid"])
        name_seq = int(row["name_seq"])

        show_stream = (i % SHOW_EVERY == 0)

        print(f"\n=== {i}/{total} PID {pid} name_seq {name_seq} ===")
        print("RAW:", raw)

        if looks_like_conference_string(raw) and looks_like_has_date(raw):
            if show_stream:
                print("LLM output (streaming):")
            parsed = parse_with_llm(raw, show_stream=show_stream)
        else:
            parsed = {
                "conf_name": normalize_conf_name(raw),
                "conf_place": "",
                "conf_dates": "",
                "note": "no date detected or skipped by heuristic",
            }

        b_day, b_month, b_year, e_day, e_month, e_year = derive_dates_from_conf_dates(
            parsed["conf_dates"]
        )
        conf_order = extract_conf_order(parsed["conf_name"])

        candidates = find_series_candidates(con, parsed["conf_name"])
        series_slug, stream_iri, series_name, series_reason = choose_series_with_llm(
            parsed["conf_name"], parsed["conf_dates"], candidates
        )

        print(
            "PARSED:",
            f"name='{parsed['conf_name']}' | place='{parsed['conf_place']}' | "
            f"dates='{parsed['conf_dates']}' | order={conf_order}",
        )
        if parsed.get("note"):
            print("NOTE:", parsed["note"])

        if series_slug or stream_iri or series_name:
            print(
                "DBLP:",
                f"slug='{series_slug}' | iri='{stream_iri}' | series='{series_name}'",
            )
            if series_reason:
                print("DBLP_REASON:", series_reason)
        else:
            print("DBLP: no series match")

        print()
        print()

        rows.append(
            {
                "pid": pid,
                "name_seq": name_seq,
                "raw_conference": raw,
                "conf_name": parsed["conf_name"],
                "conf_place": parsed["conf_place"],
                "conf_dates": parsed["conf_dates"],
                "conf_begin_date_day": b_day,
                "conf_begin_date_month": b_month,
                "conf_begin_date_year": b_year,
                "conf_end_date_day": e_day,
                "conf_end_date_month": e_month,
                "conf_end_date_year": e_year,
                "conf_order": conf_order,
                "conf_series_slug": series_slug,
                "conf_series_stream_iri": stream_iri,
                "conf_series_name": series_name,
                "conf_series_match_reason": series_reason,
                "note": parsed["note"],
            }
        )

    out = pd.DataFrame(rows)
    print("\nSample of parsed output:")
    print(out.head(20).to_string(index=False))

    write_parsed_table(con, out, "names_conference_parsed")
    out.to_csv("names_conference_parsed_sample.csv", index=False)

    con.close()
    print("\nDone. Wrote parsed data to 'names_conference_parsed' and CSV.")


if __name__ == "__main__":
    main()
```

This split gives you:

- One place to tweak regex heuristics (`regex_utils.py`).
- One place to tweak the LLM prompt and parsing (`llm_parse.py`).
- One place to adjust dblp matching and future sources (`llm_series.py`).
- A very small `pipeline.py` that just wires steps together.

You can drop these modules alongside your existing script and gradually migrate; behaviour should stay the same apart from the module boundaries.

