import random
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

def generate_recipe_prompt(recipe_block: str) -> str:
    # search for text between <TITLE> </TITLE> tokens
    # grab everything inside
    match = re.search(r"<TITLE>(.*?)</TITLE>", recipe_block, re.DOTALL)
    if not match:       # if not tag was found then return an empty string
        return ""

    # match.group(1) extracts the text captured by (.*?) which is the title
    # .strip() removes leading or trailing whitespaces, or newlines around the title
    # .lower converts the string to lowercase
    title = match.group(1).strip().lower()
    
    # picks one template string at random, replaces the {title} placeholder in the chosen template with the actual title value
    return random.choice(PROMPT_TEMPLATES).format(title=title)

