***

## 1. Download N‑Triples dump on the laptop (recommended)

From the same place you got `dblp.rdf`, also download the N‑Triples dump (`*.nt.gz`) – it is the same graph but line‑based, ideal for streaming filters.[^1][^2]

Suppose you have:

```text
dblp.nt.gz   # compressed N-Triples
```


***

## 2. Stream-filter conference series into CSV

Create `extract_conference_series.py` on your laptop:

```python
#!/usr/bin/env python3
import gzip
import csv
import re

NT_PATH = "dblp.nt.gz"
OUT_CSV = "dblp_conference_series.csv"

# N-Triples line pattern: <subject> <predicate> <object> .
# dblp schema uses rdf:type dblp:Conference for series. [web:138][web:141]
TYPE_PRED = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
CONF_OBJ = "<https://dblp.org/rdf/schema#Conference>"
LABEL_PRED = "<http://www.w3.org/2000/01/rdf-schema#label>"

uri_re = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+(.*)\s\.\s*$")


def main():
    # Two-pass streaming:
    # 1) Collect all subjects that are Conferences.
    # 2) Collect labels for those subjects.

    conference_subjects = set()

    # Pass 1: find all conference series IRIs
    with gzip.open(NT_PATH, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if TYPE_PRED in line and CONF_OBJ in line:
                m = uri_re.match(line)
                if not m:
                    continue
                subj = m.group(1)  # subject URI
                conference_subjects.add(subj)

    print(f"Found {len(conference_subjects)} conference series")

    # Pass 2: get labels for those series
    with gzip.open(NT_PATH, "rt", encoding="utf-8", errors="replace") as f, \
         open(OUT_CSV, "w", newline="", encoding="utf-8") as out_f:

        writer = csv.writer(out_f, delimiter=";")
        writer.writerow(["stream_iri", "series_name"])

        for line in f:
            if LABEL_PRED not in line:
                continue

            m = uri_re.match(line)
            if not m:
                continue

            subj = m.group(1)
            if subj not in conference_subjects:
                continue

            obj = m.group(3)  # label literal, e.g. "International Conference on X"@en
            if not obj.startswith('"'):
                continue
            # crude literal parsing: strip leading/trailing quote and optional @lang
            label = obj.split('"', 2)[^1]
            writer.writerow([subj, label])

    print(f"Wrote conference series labels to {OUT_CSV}")


if __name__ == "__main__":
    main()
```

What this does:

- **Pass 1**: finds all triples of the form
`?s rdf:type dblp:Conference` and collects the subject IRIs.[^3][^4]
- **Pass 2**: for those subject IRIs, finds `rdfs:label` triples and writes `(stream_iri, label)` to CSV.

It never materializes the whole graph; it only stores the set of conference IRIs in RAM plus the output CSV.

Run it:

```bash
python3 extract_conference_series.py
```

On a modern laptop this will take some time, but is feasible and will produce a CSV likely in the tens of MB range.

***

## 3. Move the CSV to the VM and load into DuckDB

Once you have `dblp_conference_series.csv`:

- Copy to the VM (`scp`, `rsync`, etc.).
- On the VM, in DuckDB:

```sql
CREATE TABLE dblp_conference_series AS
SELECT * FROM read_csv_auto('dblp_conference_series.csv', delim=';');
```

You now have:

- `stream_iri`: stable dblp conference series identifier.
- `series_name`: canonical label.

You can add:

- A derived `series_slug` (e.g. suffix of the IRI, like `icassp`).
- Your own `conf_series_id` surrogate key.

And then hook this into your LLM‑based normalizer as the authority table.


# ok, so i have added the table dblp_conference_series to my database kth_metadata.duckdb now

Great, that’s the key piece you needed. Now you can start wiring it into your parsing/normalization pipeline so each conference string can be mapped to a series.

Here is a minimal, concrete way to integrate it with your existing script.

***

## 1. Extend your authority table in DuckDB

Right now `dblp_conference_series` has at least:

- `stream_iri`
- `series_name`

Derive a `series_slug` (suffix of the IRI) and an internal ID:

```sql
-- Run once in kth_metadata.duckdb
ALTER TABLE dblp_conference_series
    ADD COLUMN IF NOT EXISTS series_slug TEXT;

UPDATE dblp_conference_series
SET series_slug = regexp_extract(stream_iri, '.*/([^/]+)$', 1)
WHERE series_slug IS NULL;

CREATE SEQUENCE IF NOT EXISTS conf_series_seq;

ALTER TABLE dblp_conference_series
    ADD COLUMN IF NOT EXISTS conf_series_id BIGINT;

UPDATE dblp_conference_series
SET conf_series_id = nextval('conf_series_seq')
WHERE conf_series_id IS NULL;
```

Now you have a local `conf_series_id` plus the dblp `stream_iri` and a slug.

***

## 2. Add a simple candidate lookup in Python

In your parsing script, after you have `parsed["conf_name"]` and optionally a `series_acronym` (you can heuristically take the last all‑caps token), you can ask DuckDB for a small candidate set.

Example helper:

```python
import duckdb

def find_series_candidates(con, conf_name: str, max_candidates: int = 10):
    """
    Return up to max_candidates rows from dblp_conference_series that
    could match this conference name.
    """
    name = conf_name or ""
    # crude acronym heuristic: last all-caps token of length >= 3
    import re
    tokens = re.findall(r"\b[A-Z]{3,}\b", name)
    acronym = tokens[-1] if tokens else ""

    if acronym:
        # Prefer matching on slug or acronym in label
        query = """
        SELECT conf_series_id, stream_iri, series_name
        FROM dblp_conference_series
        WHERE series_slug ILIKE ?
           OR series_name ILIKE ?
        LIMIT ?
        """
        params = [acronym.lower(), f"%{acronym}%", max_candidates]
    else:
        # Fallback: substring match on a trimmed name prefix
        short = " ".join(name.split()[:5])
        query = """
        SELECT conf_series_id, stream_iri, series_name
        FROM dblp_conference_series
        WHERE series_name ILIKE ?
        LIMIT ?
        """
        params = [f"%{short}%", max_candidates]

    return con.execute(query, params).fetchall()
```

You can call this inside your main loop after you have `parsed["conf_name"]` to get a small list of candidate series for that record.

***

## 3. Optional: let the LLM pick the series

Once you have `candidates` (say 0–10 rows), you can include them in your LLM prompt and ask:

- “Which `conf_series_id` (if any) best matches this conference string?”

You would then store the chosen `conf_series_id` in your output row alongside `conf_name`, `conf_place`, etc.

***

With this setup, you now have:

- dblp’s **conference series** as an authority table in DuckDB.
- A path from each raw `conference` string → normalized name → small candidate set → chosen `conf_series_id`.

That is enough to start experimenting with real series mapping using your existing script.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

