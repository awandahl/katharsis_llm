

***

## 1. Updated `conf2csv.py` with tqdm

Install tqdm in your venv:

```bash
(venv) pip install tqdm
```

Then use this script (still using `lxml`):

```python
#!/usr/bin/env python3
from lxml import etree
import csv
from tqdm import tqdm  # progress bar

DBLP_XML = "/root/dblp/dblp.xml"
OUT_CSV = "dblp_proceedings.csv"
TARGET_TAG = "proceedings"

# Optional: rough estimate of number of proceedings to make tqdm show %
# You can set TOTAL_PROCEEDINGS = None if you don't know.
TOTAL_PROCEEDINGS = None  # e.g. 80_000 if you measure once


def iter_proceedings(xml_path: str):
    context = etree.iterparse(
        xml_path,
        events=("end",),
        load_dtd=True,
        resolve_entities=True,
        recover=True,
        encoding="iso-8859-1",
    )

    for event, elem in context:
        if elem.tag == TARGET_TAG:
            key = elem.get("key", "")
            title_el = elem.find("title")
            year_el = elem.find("year")
            title = title_el.text if title_el is not None else ""
            year = year_el.text if year_el is not None else ""
            yield key, title, year

            # free memory for this subtree and its processed siblings
            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while parent.getprevious() is not None:
                    del parent.getparent()[^0]


def main():
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["key", "title", "year"])

        # wrap the generator with tqdm for a progress bar
        for key, title, year in tqdm(
            iter_proceedings(DBLP_XML),
            total=TOTAL_PROCEEDINGS,
            unit="proc",
            desc="Parsing dblp proceedings",
            mininterval=1.0,
        ):
            w.writerow([key, title, year])


if __name__ == "__main__":
    main()
```

Notes:

- `tqdm(iterable, total=..., unit="proc", desc="...")` wraps your generator and updates the bar on each iteration.[^3][^1]
- `TOTAL_PROCEEDINGS` is optional; if left `None`, tqdm still shows a counter and rate, just not a % or ETA.[^4]
- The memory‑cleanup logic is kept so the VM shouldn’t blow up.

***

## 2. Run with `ionice` and `nice`

From the shell:

```bash
cd /root/dblp
ionice -c2 -n7 nice -n 10 venv/bin/python conf2csv.py
```

- `nice -n 10` lowers CPU priority.
- `ionice -c2 -n7` sets best‑effort IO with lowest priority on Linux.

Together with tqdm, you now get:

- A live progress bar in the terminal.
- A process that is gentler on CPU and disk, so the VM feels less frozen.

