#!/usr/bin/env python3
import sys
from tabulate import tabulate
from collections import defaultdict

path_file = sys.argv[1]
try:
    style = sys.argv[2]
except IndexError:
    style = "simple"

with open(path_file) as f:
    lines = f.readlines()

d_result = defaultdict(lambda: defaultdict(dict))
env = None
for line in lines:
    if "export" in line:
        env = line.split("export")[1].strip()
    elif any(t in line for t in ["FAILURE", "SUCCESS"]):
        if "FAILURE" in line:
            result = "FAILURE"
        elif "SUCCESS" in line:
            result = "SUCCESS"
        type_of_concurency, argv, *_ = line[2:].split('|') # Remove ## prefix
        d_result[env][" ".join(argv.split())][type_of_concurency.strip()] = result

for name, table_data in d_result.items():
    l_of_dict = [{"commands": name, **type_} for name, type_ in table_data.items()]
    print(name)
    print(tabulate(l_of_dict, headers="keys", tablefmt=style))
    print("")
