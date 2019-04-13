#!/usr/bin/env python3

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

json_file = sys.argv[1]

with open(json_file) as data_file:
    data = json.load(data_file)

for sequence_folding in data:
    print(sequence_folding[0])