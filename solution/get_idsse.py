"""Descarga los 7 partidos de IDSSE (DFL, CC-BY 4.0) desde el espejo público de PySport en Hugging Face.

Bassek, M., Rein, R., Weber, H., Memmert, D. "An integrated dataset of spatiotemporal and event data in
elite soccer." Sci Data 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y

    .venv/bin/python solution/get_idsse.py
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "data", "external", "idsse")
BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
MATCHES = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
           "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
           "J03WR9": "DFL-COM-000002"}


def main():
    os.makedirs(DST, exist_ok=True)
    for mid, comp in MATCHES.items():
        for tmpl in ("DFL_02_01_matchinformation_{c}_DFL-MAT-{m}.xml", "DFL_04_03_positions_raw_observed_{c}_DFL-MAT-{m}.xml"):
            name = tmpl.format(c=comp, m=mid)
            path = os.path.join(DST, name)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            print("descargando", name, flush=True)
            urllib.request.urlretrieve(BASE + name, path + ".part")
            os.replace(path + ".part", path)
    print("IDSSE listo en", DST)


if __name__ == "__main__":
    main()
