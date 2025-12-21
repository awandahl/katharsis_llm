#!/usr/bin/env python3
import duckdb
import re
import pandas as pd

DB_PATH = "kth_metadata.duckdb"

# Very rough patterns – tune to your data
DATE_REGEX = re.compile(
    r"(?P<start_day>\d{1,2})\s*(?:–|-|to)?\s*(?P<end_day>\d{1,2})?\s+"
    r"(?P<month>[A-Za-zÅÄÖåäö]+)\s+(?P<year>\d{4})"
)

# Common separators between name / place / date
SEP_PATTERN = re.compile(r"\s*[,;]\s*")

def connect():
    return duckdb.connect(DB_PATH)

def fetch_conf_strings(con):
    # Adjust table/column names as needed
    df = con.execute("""
        SELECT
            PID,
            names_conference
        FROM names_conference
        WHERE names_conference IS NOT NULL
    """).fetch_df()  # standard DuckDB Python use[web:253][web:325]
    return df

def parse_conference_string(text: str):
    """
    Heuristic:
    - Find date substring with DATE_REGEX.
    - Take the part before date as "name + place".
    - If there's a last comma/semicolon, treat right-hand side as place, left as name.
    """
    if not text:
        return None, None, None

    t = text.strip()

    # Find date pattern
    m = DATE_REGEX.search(t)
    dates = None
    before = t
    if m:
        start_day = m.group("start_day")
        end_day = m.group("end_day")
        month = m.group("month")
        year = m.group("year")
        if end_day:
            dates = f"{start_day}–{end_day} {month} {year}"
        else:
            dates = f"{start_day} {month} {year}"
        before = t[:m.start()].rstrip(" ,;")

    # Split "before" into name / place using last separator
    name = before
    place = None
    # find last comma/semicolon
    m2 = list(SEP_PATTERN.finditer(before))
    if m2:
        last = m2[-1]
        left = before[:last.start()].strip()
        right = before[last.end():].strip()
        if left and right:
            name = left
            place = right

    return name or None, place or None, dates or None

def main():
    con = connect()
    df = fetch_conf_strings(con)
    print(f"Got {len(df)} conference strings")

    rows = []
    for _, row in df.iterrows():
        raw = row["names_conference"]
        name, place, dates = parse_conference_string(raw)
        rows.append(
            {
                "pid": int(row["PID"]),
                "raw": raw,
                "conf_name": name,
                "conf_place": place,
                "conf_dates": dates,
            }
        )

    out = pd.DataFrame(rows)
    # inspect some examples to see how well the heuristic works
    print(out.head(30).to_string(index=False))

    out.to_csv("conference_parsed_from_names_conference.csv", index=False)

if __name__ == "__main__":
    main()
