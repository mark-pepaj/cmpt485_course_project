import pandas as pd
import ast
import re

recipes_file = "shuffled_recipes.csv"
with open(recipes_file, 'r') as f:
    num_rows = sum(1 for line in f)

UNIT_MAP = {
    "c.": "cup",
    "cups": "cup",
    "tbsp.": "tablespoon",
    "tsp.": "teaspoon",
    "oz.": "ounce",
    "pkg.": "package",
    "lb.": "pound",
    "lbs.": "pound",
}


def parse_string_list(s):
    try:
        lst = ast.literal_eval(s)
        if not isinstance(lst, list):
            return []
        return [
                normalize_units(

def format_rows(row):    
    
with open(recipes_file, 'r', encoding="utf-8") as f:
    for line in f:
        line = line.dropna()
        
