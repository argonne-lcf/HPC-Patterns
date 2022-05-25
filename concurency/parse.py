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
        env = line.split('export')[1].strip()
    elif "./" in line:
        type_of_concurency, *argv = line.strip().split()[2:]
        profiling = False
        if argv[-1] == '--enable_profiling':
            profiling = True
            argv.pop()
    elif any(t in line for t in ["FAILURE","SUCCESS"]):
        result = line.split(':')[0]
        if profiling and result == "FAILURE":
            result = "SUCCESS*"
        d_result[env][" ".join(argv)][type_of_concurency] = result

for name, table_data in d_result.items():
    l_of_dict = [ { "commands":name, **type_} for name, type_ in table_data.items() ]
    print (name)
    print(tabulate(l_of_dict,headers="keys",tablefmt=style))
    print ('')
