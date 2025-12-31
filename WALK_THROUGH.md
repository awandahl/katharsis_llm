
Here is a walk‑through of the names_conference_parse.py, showing each code block, what it does, and how Python handles it.

***

## Imports and configuration

```python
#!/usr/bin/env python3
import duckdb
import pandas as pd
import requests
import json
import re

# ---------------- CONFIG ----------------
DB_PATH = "kth_metadata.duckdb"

MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MAX_ROWS = 200      # start small to inspect, then increase/disable
SHOW_EVERY = 1      # show streaming for every Nth row (1 = all)
# ---------------------------------------
```

- The `import` statements load external modules into the current Python process so their functions and classes become available by name:
    - `duckdb` gives access to the embedded DuckDB database engine.[^1]
    - `pandas` is imported as `pd` for DataFrames.
    - `requests` handles HTTP.
    - `json` parses and creates JSON.
    - `re` provides regular expressions.
- The constants (`DB_PATH`, `MODEL`, `OLLAMA_URL`, `MAX_ROWS`, `SHOW_EVERY`) are module‑level variables.
    - They are created when the module is first imported or executed.
    - Any function in this file can read them directly because they are in the module’s global scope.
- The `#!/usr/bin/env python3` line is a Unix shebang:
    - When you run the file as an executable, the OS uses `/usr/bin/env` to locate `python3` and run the script with it.

***

## Database connection and fetching data

```python
def connect():
    return duckdb.connect(DB_PATH)
```

- This defines a function `connect` with no parameters.
- When called, it executes `duckdb.connect(DB_PATH)`:
    - `duckdb.connect` is a function from the DuckDB module that returns a `DuckDBPyConnection` object, representing a connection to the database file.[^1]
- The `return` statement hands that connection object back to the caller.

```python
def fetch_conferences(con, limit=MAX_ROWS):
    df = con.execute(f"""
        SELECT
            pid,
            name_seq,
            conference
        FROM names_conference
        WHERE conference IS NOT NULL
        LIMIT {limit}
    """).fetch_df()  # DuckDB -> pandas
    return df
```

- `fetch_conferences` takes:
    - `con`: a DuckDB connection object.
    - `limit`: a Python integer with default `MAX_ROWS`.
- `f"""...{limit}..."""` is an f‑string:
    - The `{limit}` inside the string gets replaced with the current value of `limit` when the string is created.
- `con.execute(sql)`:
    - Sends the SQL string to DuckDB via this connection and returns a cursor‑like object.[^1]
- `.fetch_df()` on that cursor:
    - Executes the query and converts the result into a pandas `DataFrame` `df`.[^2]
- `return df` returns the DataFrame to the caller.

***

## Streaming LLM HTTP calls

```python
def stream_llm_json(prompt: str, show_stream: bool = True):
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
```

- The function signature uses type hints:
    - `prompt: str` expects a string.
    - `show_stream: bool = True` is an optional boolean, defaulting to `True`.
- `requests.post(...)`:
    - Opens an HTTP POST request to `OLLAMA_URL`.
    - The `json={...}` argument tells `requests` to JSON‑encode that dict into the request body and set `Content-Type: application/json`.[^3]
    - `stream=True` keeps the TCP connection open and allows iterating over the response as it arrives, instead of reading everything at once.[^4]
- `resp` is a `Response` object.
- `resp.raise_for_status()`:
    - Checks the HTTP status code and raises an exception if it indicates an error (4xx/5xx).[^3]

```python
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
```

- `full_text = []` creates an empty Python list to store text chunks.
- `for line in resp.iter_lines():`:
    - `iter_lines()` yields the response body split at line boundaries, as each new block of bytes arrives from the server.[^4]
    - Each `line` is a `bytes` object.
- `if not line: continue`:
    - Skips empty keep‑alive lines.
- `line.decode("utf-8")` converts bytes → Python `str`.
- `json.loads(...)` parses that string as JSON into a Python dict `data`.
- `data.get("response", "")`:
    - Returns the value for key `"response"` if it exists, otherwise `""`.
- If `show_stream` is `True` and the chunk is non‑empty:
    - `print(chunk, end="", flush=True)`:
        - `end=""` avoids adding a newline.
        - `flush=True` forces the terminal output buffer to flush immediately.
- `full_text.append(chunk)` adds each chunk to the list.
- `if data.get("done"):`:
    - When the server sends a JSON object with `"done": true`, the loop breaks.

```python
    if show_stream:
        print()  # newline after streaming
    return "".join(full_text)
```

- After the loop, if `show_stream` is `True`, a plain `print()` adds a newline.
- `"".join(full_text)` concatenates all string elements in `full_text` into one single string and returns it.

***

## Date heuristics: detecting date‑like content

```python
HAS_YEAR = re.compile(r"\b(19|20)\d{2}\b")
HAS_MONTH = re.compile(
    r"\b("
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|October|November|December"
    r")\b",
    re.IGNORECASE,
)
```

- `re.compile(pattern, flags)` compiles a regex once, so it is faster to reuse many times.
- `HAS_YEAR`:
    - `\b` means a word boundary.
    - `(19|20)\d{2}` matches years 1900–2099.
- `HAS_MONTH`:
    - Wrapped pattern `r"...|..."` is concatenated across multiple string literals.
    - Matches abbreviations and full month names, ignoring case.

```python
def looks_like_has_date(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return bool(HAS_YEAR.search(t) or HAS_MONTH.search(t))
```

- `if not text:` returns `False` for `None`, `""`, etc.
- `str(text)` ensures `text` is a string even if originally a non‑string type.
- `.strip()` removes leading/trailing whitespace.
- `HAS_YEAR.search(t)` and `HAS_MONTH.search(t)` perform regex matching anywhere in the string.
- `bool(...)` converts the match object or `None` into `True`/`False`.

***

## Conference‑like string heuristic

```python
MIN_LEN_FOR_LLM = 10
MAX_LEN_FOR_LLM = 400  # avoid huge blobs (abstracts, HTML, etc.)

HTML_TAG_RE = re.compile(r"<[^>]+>")
MANY_DIGITS_RE = re.compile(r"\d{6,}")  # long numeric junk
```

- These constants constrain acceptable text lengths and patterns.
- `HTML_TAG_RE` looks for `<something>` patterns.
- `MANY_DIGITS_RE` detects long digit sequences (`\d{6,}` means 6 or more digits).

```python
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
```

- The function converts any non‑`None` input to string, strips it, and checks conditions:
    - Uses `len(t)` to enforce length boundaries.
    - Uses `.search()` on the compiled regex patterns.
- Returning `True` or `False` lets other code decide whether to call the LLM.

***

## Place normalization

```python
def normalize_place(place: str) -> str:
    """
    - Remove leading/trailing space.
    - If the place is ALL CAPS (letters only), downcase then title-case.
    - Otherwise leave as is (LLM is expected to add correct å/ä/ö/ü where it knows them).
    """
    if not place:
        return place
    p = str(place).strip()
    letters = "".join(ch for ch in p if ch.isalpha())
    if letters and letters.isupper():
        return p.lower().title()
    return p
```

- `if not place:` returns `place` as‑is for `None`/empty.
- `p = str(place).strip()` ensures string type and trimmed whitespace.
- `"".join(ch for ch in p if ch.isalpha())`:
    - A generator expression `ch for ch in p if ch.isalpha()` iterates characters, selecting only letters.
    - `join` concatenates them into `letters`.
- `letters.isupper()` is `True` if all alphabetic characters are uppercase.
- `p.lower().title()`:
    - `.lower()` converts the entire string to lowercase.
    - `.title()` capitalizes the first letter of each word.
- Otherwise, it returns `p` unchanged.

***

## Conference name normalization

```python
SMALL_WORDS = {
    "and", "or", "of", "on", "in", "for", "to", "the", "a", "an", "at", "by", "with",
}

ACRONYM_OVERRIDES = {
    "eccomas": "ECCOMAS",
    # "ifac": "IFAC",
    # "icas": "ICAS",
}
```

- `SMALL_WORDS` is a `set` of lowercased small words to keep lowercase inside titles.
- `ACRONYM_OVERRIDES` is a `dict` mapping lowercased acronyms to the desired exact casing.

```python
def normalize_conf_name(name: str) -> str:
    """
    Normalize conference name to title-style capitalization:
    ...
    """
    if not name:
        return name
    text = str(name).strip()
```

- Checks for falsy `name`, returns as‑is.
- Normalizes to stripped string `text`.

```python
    tokens = re.split(r"(\s+)", text)  # keep spaces
    result = []
    start_of_segment = True
```

- `re.split(r"(\s+)", text)`:
    - Splits on whitespace, but because the regex has parentheses, the delimiters (spaces) are kept as separate tokens.
    - Example: `"A  B"` → `["A", "  ", "B"]`.
- `result` accumulates transformed tokens.
- `start_of_segment` is a state flag for “start of title segment”.

```python
    for tok in tokens:
        if tok.isspace():
            result.append(tok)
            continue
```

- `str.isspace()` checks if the token is entirely whitespace.
- Whitespace tokens are preserved as‑is.

```python
        word = tok
        trailing = ""
        m = re.match(r"^([A-Za-z0-9]+)(.*)$", tok)
        if m:
            word, trailing = m.group(1), m.group(2)

        lower_word = word.lower()
```

- Initializes `word` and `trailing`.
- `re.match` tries to split the token into:
    - A leading alphanumeric chunk `([A-Za-z0-9]+)`.
    - The rest `(.*)` (could be punctuation).
- `m.group(1)` is the main word, `m.group(2)` is the trailing part.
- `lower_word` is a lowercase version for case‑insensitive checks.

```python
        # Explicit acronym overrides first
        if lower_word in ACRONYM_OVERRIDES:
            new_word = ACRONYM_OVERRIDES[lower_word]
        # Preserve clear acronyms: all caps and length > 1
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
```

- If the lowercase word matches an override, use that exact mapped value.
- Else, if the original word is all uppercase and longer than one character, keep it (acronym).
- Otherwise:
    - If at the start of a segment, capitalize first letter and lowercase rest.
    - Else if the word is in `SMALL_WORDS`, use the lowercase form.
    - Else use title‑style capitalization again.

```python
        result.append(new_word + trailing)
        # Update segment flag: after a colon, next word starts a new segment
        if ":" in tok:
            start_of_segment = True
        else:
            start_of_segment = False

    return "".join(result)
```

- `new_word + trailing` reattaches any punctuation.
- If the original token contains a colon, the next word will be treated as a new title segment.
- `"".join(result)` concatenates all transformed tokens (including spaces) into the final normalized string.

***

## Parsing normalized date ranges

```python
ISO_RANGE_RE = re.compile(
    r"""
    ^
    (?P<start>\d{4}(?:-\d{2}(?:-\d{2})?)?)      # 2025 or 2025-07 or 2025-07-13
    (?:\s*/\s*
       (?P<end>\d{4}(?:-\d{2}(?:-\d{2})?)?)     # same patterns
    )?
    $
    """,
    re.VERBOSE,
)
```

- A verbose regex to match:
    - `start`: a date `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`.
    - Optional `/` followed by `end` in the same format.
- `re.VERBOSE` allows spaces and comments inside the pattern.

```python
def parse_iso_like_date(s: str):
    """Parse YYYY[-MM[-DD]] into (year, month, day) or (None, None, None)."""
    if not s:
        return (None, None, None)
    parts = s.split("-")
    try:
        year = int(parts[^0])
    except ValueError:
        return (None, None, None)
    month = int(parts[^1]) if len(parts) >= 2 else None
    day = int(parts[^2]) if len(parts) >= 3 else None
    return (year, month, day)
```

- Splits on `-` and tries to parse the first part as an integer `year`.
- If parsing fails, returns three `None`s.
- Parses month and day only if corresponding parts exist.
- Uses conditional expressions (`X if cond else Y`) for brevity.

```python
def derive_dates_from_conf_dates(conf_dates: str):
    """
    Expected formats for conf_dates (normalized by LLM):
      ...
    """
    if not conf_dates:
        return (None, None, None, None, None, None)

    m = ISO_RANGE_RE.match(conf_dates.strip())
    if not m:
        return (None, None, None, None, None, None)
```

- Falsy `conf_dates` immediately yields six `None`s.
- `conf_dates.strip()` removes whitespace; `.match()` tries to match the full pattern.
- If no match, returns all `None`.

```python
    start_raw = m.group("start")
    end_raw = m.group("end") or start_raw

    sy, sm, sd = parse_iso_like_date(start_raw)
    ey, em, ed = parse_iso_like_date(end_raw)

    # Return in your preferred order (day, month, year)
    return (sd, sm, sy, ed, em, ey)
```

- Named groups `start` and `end` are accessed with `.group("name")`.
- `m.group("end") or start_raw`:
    - If `end` is `None`, `end_raw` falls back to `start_raw`.
- The function returns dates in day‑month‑year order for convenience.

***

## Extracting conference order (series number)

```python
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    ...
    "thirtieth": 30,
}

ORDINAL_NUMBER_RE = re.compile(r"\b(\d+)[’']?(st|nd|rd|th)\b", re.IGNORECASE)
```

- `ORDINAL_WORDS` maps english ordinal words to ints.
- `ORDINAL_NUMBER_RE` matches things like `5th`, `21st`, with optional apostrophe and case‑insensitive suffix (`ST`, `st`, etc.).

```python
def extract_conf_order(text: str):
    """
    Extracts the conference order number (e.g. 5 for 'Fifth International...').
    Returns int or None.
    """
    if not text:
        return None
    t = str(text)
```

- Returns `None` for falsy inputs.
- Converts to string for safety.

```python
    # 1) Look for numeric ordinals like "5th", "21st"
    m = ORDINAL_NUMBER_RE.search(t)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
```

- `search` finds the first match anywhere.
- `m.group(1)` is the numeric part; `int()` converts to an integer.
- `try/except` catches unexpected parse errors.

```python
    # 2) Look for word ordinals like "Fifth", "Twenty-First"
    words = re.findall(r"[A-Za-z\-]+", t)
    for w in words:
        key = w.lower()
        if key in ORDINAL_WORDS:
            return ORDINAL_WORDS[key]

    return None
```

- `re.findall` returns all substrings matching the pattern (letters and hyphens).
- Each word is lowercased and checked against `ORDINAL_WORDS`.
- If nothing is found, `None` is returned.

***

## LLM parsing with cache

```python
_llm_cache = {}
```

- `_llm_cache` is a module‑level dictionary used as an in‑memory cache:
    - Keys: raw conference strings.
    - Values: parsed result dictionaries.

```python
def parse_with_llm(conf_string: str, show_stream: bool = True):
    """
    Ask LLM to classify the string into name/place/dates.
    Returns dict with:
      conf_name, conf_place, conf_dates (normalized text), note.
    Numeric begin/end date fields are derived in Python.
    """

    if conf_string is None:
        return {
            "conf_name": "",
            "conf_place": "",
            "conf_dates": "",
            "note": "",
        }
```

- For `None`, returns a dict with empty values.

```python
    # Simple exact-string cache to avoid recomputing repeated values
    if conf_string in _llm_cache:
        cached = _llm_cache[conf_string].copy()
        # Do not stream cached output
        return cached
```

- Checks if the string is already in `_llm_cache`.
- `.copy()` creates a shallow copy of the cached dict so callers can mutate it without affecting the cache.

```python
    instruction = """
You are cleaning conference metadata.
...
"""
```

- `instruction` is a multi‑line triple‑quoted string literal containing the instructions for the LLM.
- Python retains newlines and spaces inside triple quotes.

```python
    prompt = instruction + f"\n\nRaw conference string:\n{conf_string}\n\nJSON:"

    text = stream_llm_json(prompt, show_stream=show_stream)
```

- Concatenates `instruction` with specific input to form `prompt`.
- Calls `stream_llm_json`, which returns the full textual response from the model.

```python
    # Try to extract a single JSON object from the output
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
```

- Uses string methods `.find` and `.rfind` to locate the first `{` and last `}`.
- If these indices are invalid, constructs a fallback dict and caches it.

```python
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
```

- `text[start : end + 1]` uses slice syntax to extract a substring.
- `json.loads` attempts to parse JSON into a Python dict.
- On `JSONDecodeError`, returns and caches another fallback.

```python
    result = {
        "conf_name": normalize_conf_name(str(obj.get("conf_name", "") or "")),
        "conf_place": normalize_place(obj.get("conf_place", "") or ""),
        "conf_dates": str(obj.get("conf_dates", "") or ""),
        "note": str(obj.get("note", "") or ""),
    }
    _llm_cache[conf_string] = result
    return result
```

- `obj.get("key", "")` grabs the value or returns `""` if missing.
- `or ""` ensures that falsy values (like `None`) become empty strings.
- `normalize_conf_name` and `normalize_place` are applied.
- The result dict is cached and returned.

***

## Main orchestration logic

```python
def main():
    con = connect()
    df = fetch_conferences(con)
    total = len(df)
    print(f"Fetched {total} conference rows for parsing")
```

- `main` is the entry‑point function.
- `connect()` returns a DuckDB connection.
- `fetch_conferences(con)` returns a pandas `DataFrame` `df`.
- `len(df)` gives the number of rows.
- `print(...)` outputs a formatted string via f‑string.

```python
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        raw = row["conference"]
        pid = int(row["pid"])
        name_seq = int(row["name_seq"])

        show_stream = (i % SHOW_EVERY == 0)

        print(f"\n=== {i}/{total} PID {pid} name_seq {name_seq} ===")
        print("RAW:", raw)
```

- `rows = []` will hold output rows as dicts.
- `df.iterrows()` yields `(index, row)` pairs:
    - `index` is the original index label.
    - `row` is a pandas `Series` object.
- `enumerate(..., start=1)` gives a 1‑based counter `i`.
- `(_, row)` ignores the DataFrame index.
- `row["conference"]` accesses the column value by label.
- Casting to `int` ensures numeric types.
- `show_stream = (i % SHOW_EVERY == 0)` uses modulo to set streaming frequency.

```python
        # Decide whether to use LLM at all
        if looks_like_conference_string(raw) and looks_like_has_date(raw):
            if show_stream:
                print("LLM output (streaming):")
            parsed = parse_with_llm(raw, show_stream=show_stream)
        else:
            # fast path: no obvious date or string looks like garbage
            parsed = {
                "conf_name": normalize_conf_name(raw),
                "conf_place": "",
                "conf_dates": "",
                "note": "no date detected or skipped by heuristic",
            }
```

- Conditional executes LLM parsing only for “reasonable” conference strings with date hints.
- `parsed` is always a dict after this block, either from the LLM or the fallback.

```python
        # Derive numeric dates from normalized conf_dates
        b_day, b_month, b_year, e_day, e_month, e_year = derive_dates_from_conf_dates(
            parsed["conf_dates"]
        )

        # Extract conference order from the normalized conference name
        conf_order = extract_conf_order(parsed["conf_name"])
```

- Calls helper functions to derive:
    - Numeric begin/end dates.
    - `conf_order` from the conference name.
- Tuple unpacking assigns each returned value to its own variable.

```python
        print(
            "PARSED:",
            f"name='{parsed['conf_name']}' | place='{parsed['conf_place']}' | "
            f"dates='{parsed['conf_dates']}' | order={conf_order}",
        )
        if parsed.get("note"):
            print("NOTE:", parsed["note"])
```

- Uses multiple arguments in `print`, which are separated by spaces automatically.
- f‑strings embed dict lookups and variable values.

```python
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
                "note": parsed["note"],
            }
        )
```

- Appends a Python dict representing one output row to the `rows` list.
- Keys are column names; values come from variables and `parsed`.

```python
    out = pd.DataFrame(rows)
    print("\nSample of parsed output:")
    print(out.head(20).to_string(index=False))
```

- `pd.DataFrame(rows)` builds a DataFrame from a list of dicts:
    - Keys become column names.
    - Dict entries become row values.
- `out.head(20)` takes the first 20 rows.
- `.to_string(index=False)` formats them as a table without the index for printing.

```python
    # Write to DuckDB as a new table
    con.execute("DROP TABLE IF EXISTS names_conference_parsed")
    con.sql("CREATE TABLE names_conference_parsed AS SELECT * FROM out")  # pandas->DuckDB
```

- `con.execute(...)` runs SQL that drops the table if it exists.
- `con.sql("CREATE TABLE ... AS SELECT * FROM out")`:
    - DuckDB’s Python integration can treat `out` as a pandas DataFrame in the current process, thanks to “replacement scans.”[^2]
    - The query materializes `out` into the DuckDB table `names_conference_parsed`.

```python
    out.to_csv("names_conference_parsed_sample.csv", index=False)

    con.close()
    print("\nDone. Wrote parsed data to 'names_conference_parsed' and CSV.")
```

- `out.to_csv(...)` writes the DataFrame to a CSV file.
- `con.close()` closes the database connection.
- Final `print` informs the user.

```python
if __name__ == "__main__":
    main()
```

- `__name__` is a special module variable:
    - When the file is run as a script, `__name__ == "__main__"`.
    - When imported as a module, `__name__` is the module’s name.
- This idiom ensures `main()` runs only when executed directly, not when imported.

***

