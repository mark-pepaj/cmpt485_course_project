import csv

with open("recipes_data.csv", newline="", encoding="utf-8", buffering=1024*1024) as f, open("recipes.csv", 'w', newline="") as w:
    r = csv.reader(f)
    wtr = csv.writer(w)
    
    for i, row in enumerate(r):
        if len(row) < 3:
            continue
        wtr.writerow((row[0], row[1], row[2]))

