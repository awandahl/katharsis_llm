#!/usr/bin/env python3
import duckdb
import re
import pandas as pd

DB_PATH = "kth_metadata.duckdb"

# --- heuristics: adjust to your schema/text conventions -------------
# Example assumptions:
# - Table: pub
# - Fields:
#   * PublicationType (e.g. 'Conference paper', 'Conference contribution')
#   * PublicationName  (often the conference or series name)
#   * Place            (city, country, sometimes "City, Country")
#   * ConferenceInfo   (free text like "Stockholm, Sweden, 12–15 June 2018")
#   * Year
# --------------------------------------------------------------------

def connect():
    return duckdb.connect(DB_PATH)

def fetch_conference_rows(con):
    query = """
    SELECT
        PID,
        Year,
        PublicationType,
        PublicationName,
        Place,
        ConferenceInfo,
        Title
    FROM pub
    WHERE
        PublicationType ILIKE '%conference%'
    """
    # Standard DuckDB Python API usage[web:253][web:325]
    df = con.execute(query).fetch_df()
    return df

# very simple date pattern (European style); extend as needed
DATE_REGEX = re.compile(
    r"(?P<start_day>\d{1,2})\s*(?:–|-|to)?\s*(?P<end_day>\d{1,2})?\s+"
    r"(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})"
)

def parse_location_and_dates(place, confinfo):
    """
    Return (location, dates_string) from Place + ConferenceInfo heuristically.
    """
    text = " ".join([place or "", confinfo or ""]).strip()

    # crude location guess: first "City, Country" fragment
    location = None
    if place:
        location = place.strip()

    # try to pull dates
    dates = None
    m = DATE_REGEX.search(text)
    if m:
        start_day = m.group("start_day")
        end_day = m.group("end_day")
        month = m.group("month")
        year = m.group("year")
        if end_day:
            dates = f"{start_day}–{end_day} {month} {year}"
        else:
            dates = f"{start_day} {month} {year}"

    return location, dates

def main():
    con = connect()
    df = fetch_conference_rows(con)
    print(f"Got {len(df)} conference-like records")

    records = []
    for _, row in df.iterrows():
        loc, dates = parse_location_and_dates(
            row.get("Place"), row.get("ConferenceInfo")
        )
        records.append(
            {
                "pid": int(row["PID"]),
                "year": int(row["Year"]) if row["Year"] is not None else None,
                "title": row.get("Title"),
                "conference_name": row.get("PublicationName"),
                "location": loc,
                "dates": dates,
            }
        )

    out_df = pd.DataFrame(records)
    # Show a few examples
    print(out_df.head(20).to_string(index=False))

    # Optionally write to CSV for further processing
    out_df.to_csv("conference_metadata_extracted.csv", index=False)

if __name__ == "__main__":
    main()
