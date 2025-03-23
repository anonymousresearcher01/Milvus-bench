#!/bin/bash
set -e

sudo -v

NUM=1000000
# sudo sysctl -w vm.block_dump=1

# Experiments
python run_insert_expr.py --num "${NUM}"
python run_index_build_expr.py --num "${NUM}"
python run_index_load_expr.py --num "${NUM}"
python run_search_expr.py --num "${NUM}"

# sudo sysctl -w vm.block_dump=0