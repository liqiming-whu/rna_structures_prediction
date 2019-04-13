#!/bin/bash
src=$1
vienna=$2

mkdir vienna
cp targets.json vienna
cd vienna
$src/eval_vienna/print_targets_json.py ./targets.json | $vienna/src/bin/RNAfold -o
~/Programming/Uni/DeepLearning/lasagne/src/eval_vienna/parse_vienna_output.py RNAfold_output.fold targets.json .
