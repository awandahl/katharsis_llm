import json
import duckdb
import requests

DB_PATH = "kth_metadata.duckdb"
MODEL = "llama3.2:1b"

con = duckdb.connect(DB_PATH)

# Only rows that actually have an ORCID
df = con.execute("""
    SELECT
      pid,
      pub_title,
      pub_year,
      pub_type,
      kth_author_id,
      given_name,
      family_name,
      author_role,
      orcid,
      org_id,
      affiliation_free,
      affiliations_full,
      hsv_code,
      subject_topic
    FROM author_orcid_candidates
    WHERE orcid IS NOT NULL AND orcid <> ''
    LIMIT 20
""").fetchdf()

for _, row in df.iterrows():
    rec = row.to_dict()

    diva_name = f\"{rec['family_name']}, {rec['given_name']}\" if rec['family_name'] else rec['given_name']

    prompt = f\"\"\"You are helping a university library check metadata quality for authors and ORCID iDs.

Publication:
- PID: {rec['pid']}
- Title: {rec['pub_title']}
- Year: {rec['pub_year']}
- Type: {rec['pub_type']}

KTH/DiVA author metadata:
- Name: {diva_name}
- KTH author_id: {rec['kth_author_id']}
- Role: {rec['author_role']}
- Org info: org_id={rec['org_id']}, affiliation='{rec['affiliation_free']}', affiliations_full='{rec['affiliations_full']}'
- Subjects: HSV code={rec['hsv_code']}, topic='{rec['subject_topic']}'

ORCID metadata:
- ORCID iD: {rec['orcid']}

Task:
- Based ONLY on this information, decide if the KTH author and the ORCID iD likely belong to the same person.
- This is for improving catalog metadata quality, not for identifying or tracking individuals.
- Answer strictly in JSON of the form:
{{
  "decision": "MATCH" | "POSSIBLE_MATCH" | "MISMATCH",
  "reason": "1–3 sentence explanation"
}}
Do not add any other text.
\"\"\"  # noqa: E501

    resp = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    answer = resp.json()["response"].strip()

    obj = json.loads(answer.splitlines()[0])

    print(f"PID {rec['pid']} | {diva_name} | ORCID {rec['orcid']}")
    print(obj)
