#!/bin/bash
set -e

sudo -v

NUM=1000000
# sudo sysctl -w vm.block_dump=1

# Experiments
python run_insert_vectors_expr.py --num "${NUM}"
python run_build_index_expr.py --num "${NUM}"
python run_load_index_load_expr.py --num "${NUM}"
python run_search_vectors_expr.py --num "${NUM}"

# sudo sysctl -w vm.block_dump=0
