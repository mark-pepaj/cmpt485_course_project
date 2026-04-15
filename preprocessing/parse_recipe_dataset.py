import subprocess
import pandas as pd
import ast
import re
import os

recipes_file = "shuffled_recipes.csv"
prompts_file = "prompts.txt"

UNIT_MAP = {
    "c.": "cup",
    "tbsp.": "tablespoon",
    "tsp.": "teaspoon",
    "oz.": "ounce",
    "pkg.": "package",
    "lb.": "pound",
    "lbs.": "pound",
}


def clean_text(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def parse_list(s):
    try:
        lst = ast.literal_eval(s)
        if not isinstance(lst, list):
            return []
        return [
                normalize_units(clean_text(x))
                for x in lst
                if isinstance(x, str) and x.strip()
                ]
    except:
        return []


def load_prompts(path):
    groups = []
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append(line)
    if current:
        groups.append(current)
    return groups

def normalize_units(text):
    #text = text.lower()

    # match patterns like "1 c. milk"
    pattern = r"(\d+[\d\/\.\s]*)\s*([a-zA-Z\.]+)\s*(.*)"

    m = re.match(pattern, text)
    if not m:
        return text.strip()

    qty, unit, rest = m.groups()

    unit = unit.strip().lower()
    unit = UNIT_MAP.get(unit, unit)

    return f"{qty.strip()} {unit} {rest.strip()}".strip()


def format_recipes(row, prompt):
    title = clean_text(row[0])#"title"])

    ingredients = parse_list(row[1])#"ingredients"])
    directions = parse_list(row[2])#"directions"])

    ingredients_block = "\n".join(ingredients)
    directions_block = " ".join(directions)

    return (
            "<SOS>\n"
            "<PROMPT>" + clean_text(prompt) + "</PROMPT>\n\n"
            "<TITLE>" + title + "</TITLE>\n\n"
            "<INGREDIENTS>\nIngredients:\n" + ingredients_block + "\n</INGREDIENTS>\n\n"
            "<DIRECTIONS>\nDirections:\n" + directions_block + "\n</DIRECTIONS>\n"
            "<EOS>\n"
            )


prompt_groups = load_prompts(prompts_file)
prompt_idx = 0

total_recipes = 0




length = 0
sets = ["training", "validation", "testing"]
for s in sets:
    output_file = s + ".txt"
    open(output_file, "w").close()
    chunks = pd.read_csv(
            recipes_file,
            nrows=10,
            #usecols=["title", "ingredients", "directions"],
            dtype=str,
            chunksize=1)

    for chunk in chunks:

        chunk = chunk.dropna()#subset=['title', 'ingredients', 'directions'])
        length += len(chunk)

        texts = []
        
        for _, row in chunk.iterrows():
            if prompt_idx >= len(prompt_groups):
                break

            prompts = prompt_groups[prompt_idx]
            prompt_idx += 1

            for p in prompts:
                texts.append(format_recipes(row, p))

        with open(output_file, "a", encoding="utf-8") as f:
              f.write("".join(texts))

#subprocess.run(["rm", "recipes_data.csv"])
#subprocess.run(["clear"])
#print(f"Number of recipes: {length}")
