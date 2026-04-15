import random
import sys
import re

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


def generate_prompt(title):
    match = re.search(r"<TITLE>(.*?)</TITLE>", title, re.DOTALL)
    if not match:       
        return ""
    return random.choice(PROMPT_TEMPLATES).format(title=title)


def format_recipes(row):
    title = row["title"]

    prompt = generate_prompt(title)
    
    ingredients = row["ingredients"]
    directions = row["directions"]

    ingredients_block = "\n".join(ingredients)
    directions_block = " ".join(directions)

    return (
            "<SOS>\n"
            "<PROMPT>" + prompt + "</PROMPT>\n\n"
            "<TITLE>" + title + "</TITLE>\n\n"
            "<INGREDIENTS>\nIngredients:\n" + ingredients_block + "\n</INGREDIENTS>\n\n"
            "<DIRECTIONS>\nDirections:\n" + directions_block + "\n</DIRECTIONS>\n"
            "<EOS>\n"
            )

num_rows = sys.argv[1]

input_file = "normalized_recipes.csv"
n1 = int(0.2 * num_rows)
n2 = int(0.6 * num_rows)

out_train = open("testing.txt", 'w', encoding="utf-8")
out_val = open("validation.txt", 'w', encoding="utf-8")
out_test = open("testing.txt", 'w', encoding="utf-8")

# Xtr = data[:n1]
# Xval = data[n1:n2]
# Xte = data[n2:] 

with open(input_file, 'r', encoding="utf-8", newline="") as fin: 
    reader = csv.DictReader(fin)

    for i, row in enumerate(reader):
        text = format_recipes(row)
        if i < n1:
            out_train.write("".join(text))
        elif i >= n1 and i < n2:
            out_val.write("".join(text))
        else:
            out_test.write("".join(text))


out_train.close()
out_val.close()
out_test.close()

