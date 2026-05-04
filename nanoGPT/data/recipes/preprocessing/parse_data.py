import pandas as pd
import csv
import ast
import re

input_file = "shuffled.csv"
output_file = "normalized.csv"

"""
chunks = pd.read_csv(input_file,
                     dtype=str,
                     chunksize=100000,
                     engine="python",
                     on_bad_lines="skip"
                     )
"""

#for chunk in chunks:
#    chunk = chunk.dropna(subset=["title", "ingredients", "directions"])
    

def normalize_units(text):

    pattern = r"(\d+[\d\/\.\s]*)\s*([a-zA-Z\.]+)\s*(.*)"

    m = re.match(pattern, text)
    if not m:
        return text.strip()

    qty, unit, rest = m.groups()

    unit_key = unit.strip().lower().rstrip(".")
    unit = UNIT_MAP.get(unit_key, unit_key)

    return f"{qty.strip()} {unit} {rest.strip()}".strip()


    for original, normalized in UNIT_MAP.items():
        text = re.sub(rf"\b{re.escape(original)}\b", normalized, text)
    return text


def normalize_whitespace(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def parse_list_column(s):
    try:
        lst = ast.literal_eval(s)
        if not isinstance(lst, list):
            return []
        return [
                normalize_whitespace(x)
                for x in lst
                if isinstance(x, str) and x.strip()
                ]
    except:
        return []



"""
chunks = pd.read_csv(
    input_file,
    dtype=str,
    chunksize=50000,
    engine="python",
    on_bad_lines="skip"
    )
"""
with open(input_file, 'r', encoding="utf-8", newline="") as fin, open(output_file, 'w', encoding="utf-8", newline="") as fout:
    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
    
    writer.writeheader()

    for row in reader:
        row.pop(None, None)
        ingredients = parse_list_column(row.get("ingredients", ""))
        directions =  parse_list_column(row.get("directions", ""))

        row["ingredients"] = "\n".join(ingredients)
        row["directions"] = "\n".join(directions)

        writer.writerow(row)
