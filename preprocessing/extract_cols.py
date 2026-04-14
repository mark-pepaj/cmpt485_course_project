import csv


with open("recipes_data.csv", newline="") as f, open("recipes.csv", 'w', newline="") as w:
    r = csv.reader(f)
    wtr = csv.writer(w)

    for row in r:
        wtr.writerow(row[:3])
        wtr.writerow(row[:3])
        wtr.writerow(row[:3])

