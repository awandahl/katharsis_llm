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
            label = obj.split('"', 2)[1]
            writer.writerow([subj, label])

    print(f"Wrote conference series labels to {OUT_CSV}")


if __name__ == "__main__":
    main()
