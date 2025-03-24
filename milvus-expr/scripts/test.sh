#!/bin/bash
set -e

sudo -v

NUM=100000
# sudo sysctl -w vm.block_dump=1

# Experiments
python run_insert_vectors_expr.py --num "${NUM}"
sleep 3
python run_build_index_expr.py --num "${NUM}"
sleep 3
python run_load_index_expr.py --num "${NUM}"
sleep 3
python run_search_vectors_expr.py --num "${NUM}"

# sudo sysctl -w vm.block_dump=0
