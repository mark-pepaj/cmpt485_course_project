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

    return (
            "<SOS>\n"
            "<PROMPT>" + prompt + "</PROMPT>\n\n"
            "<TITLE>" + title + "</TITLE>\n\n"
            "<INGREDIENTS>\n" + ingredients + "\n</INGREDIENTS>\n\n"
            "<DIRECTIONS>\n" + directions + "\n</DIRECTIONS>\n"
            "<EOS>\n"
            )

input_file = "normalized.csv"
output_file = "input.txt"

# define the number of rows for a subset of the recipes
num_rows = 1000

# uncomment to get the total number of recipes
"""
num_rows = 0
with open(input_file, 'r', encoding="utf-8", buffering=1024*1024) as f:
    for line in f:
        num_rows += 1
"""


chunks = pd.read_csv(
        input_file,
        nrows = num_rows,
        usecols=["title", "ingredients", "directions"],
        dtype=str,
        chunksize=10)


with open(output_file, "a", encoding="utf-8") as f:
    for chunk in chunks:
        for row in chunk.itertuples(index=False):
            row_dict = {
                    "title": row.title,
                    "ingredients": row.ingredients,
                    "directions": row.directions
                    }
            if not all(isinstance(v, str) for v in row_dict.values()):
                continue           
            text = format_recipes(row_dict)
            f.write("".join(text))

 
