import random
import pandas as pd
import sys
import re


def generate_prompt(title):
    PROMPT_TEMPLATES = [
            "Show me a {title} recipe",
            "How do I make {title}?",
            "Give me a {title} recipe",
            "What are the steps to making {title}?",
            "How to make {title}?",
            "Show me how to prepare {title}",
            "{title} recipe",
            "Recipe for {title}",
            ]

    return random.choice(PROMPT_TEMPLATES).format(title=title)


def format_recipes(row):
    title = row["title"]

    prompt = generate_prompt(title)
    
    ingredients = row["ingredients"]
    directions = row["directions"]

    #ingredients_block = "\n".join(ingredients)
    #directions_block = " ".join(directions)

    return (
            "<SOS>\n"
            "<PROMPT>" + prompt + "</PROMPT>\n\n"
            "<TITLE>" + title + "</TITLE>\n\n"
            "<INGREDIENTS>\n" + ingredients + "\n</INGREDIENTS>\n\n"
            "<DIRECTIONS>\n" + directions + "\n</DIRECTIONS>\n"
            "<EOS>\n"
            )

input_file = "normalized.csv"
out_train = "training.txt"
out_val = "validation.txt"
out_test = "testing.txt"

"""
num_rows = 0
with open(input_file, 'r', encoding="utf-8", buffering=1024*1024) as f:
    for line in f:
        num_rows += 1
"""

num_rows = 30
n1 = int(0.2 * num_rows)
n2 = int(0.6 * num_rows)
i = 0

# Xtr = data[:n1]
# Xval = data[n1:n2]
# Xte = data[n2:] 


chunks = pd.read_csv(
        input_file,
        nrows = 30,
        usecols=["title", "ingredients", "directions"],
        dtype=str,
        chunksize=10)


with open(out_train, "a", encoding="utf-8") as f_train, open(out_val, "a", encoding="utf-8") as f_val, open(out_test, "a", encoding="utf-8") as f_test:
    for chunk in chunks:
        for row in chunk.itertuples(index=False):
            row_dict = {
                    "title": row.title,
                    "ingredients": row.ingredients,
                    "directions": row.directions
                    }
            text = format_recipes(row_dict)
            
            if i < n1:
                with open(out_train, "a", encoding="utf-8") as f:
                    f.write("".join(text))
            elif i >= n1 and i < n2:
                with open(out_val, "a", encoding="utf-8") as f:
                    f.write("".join(text))
            else:
                with open(out_test, "a", encoding="utf-8") as f:
                    f.write("".join(text))
            i += 1

