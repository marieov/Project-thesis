
import pandas as pd
import csv

print("start!")
lines = list()
remove = []

with open('features_f.csv', 'r') as read_file:
    print("opened")
    reader = csv.reader(read_file)
    print("read)")
    for row_number, row in enumerate(reader, start=1):
        if "nan" not in row[4]:
            lines.append(row)


print("write)")
with open('features_f_no_nans_2feat.csv', 'w') as write_file:
    writer = csv.writer(write_file)
    writer.writerows(lines)
