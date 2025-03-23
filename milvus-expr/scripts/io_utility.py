import os
from typing import Dict, List, Set

import pandas as pd
from tabulate import tabulate

insert_vector_phase: Set[str] = {"load_data", "prepare_collection", "flush_collection", "insert_vectors", "sync_disk"}
build_index_phase: Set[str] = {"build_index", "sync_disk"}
load_index_phase: Set[str] = set()
search_vector_phase: Set[str] = set()
supported_phases: Set[str] = insert_vector_phase | build_index_phase | load_index_phase | search_vector_phase


def load_io_stats(expr: str = None) -> Dict[str, List[Dict[str, int]]]:
    """Load I/O information reading I/O stat file"""
    io_stats = {}
    io_dir = f"../result_stat/io_monitoring/{expr}" if expr else "../result_stat/io_monitoring"

    for filename in os.listdir(io_dir):
        # NOTE(Dhmin): These file name rule depends on how to store I/O information file when monitoring I/O (See io_monitor.sh)
        if filename.startswith("io_stats_") and filename.endswith(".csv") and not filename.endswith("summary.csv"):
            phase = filename.replace("io_stats_", "").replace(".csv", "")

            # NOTE(Dhmin): sanity check
            if phase not in supported_phases:
                raise NotImplementedError(f"The {phase} has not been supported yet.")

            df = pd.read_csv(os.path.join(io_dir, filename))

            stats: List[Dict[str, int]] = []
            for _, row in df.iterrows():
                # NOTE(Dhmin): These keys of row are used when storing collected I/O information (See io_monitor.sh)
                stats.append(
                    {
                        "device": row["device"],
                        "read_operations": row["read_operations"],
                        "read_mb": row["read_mb"],
                        "write_operations": row["write_operations"],
                        "write_mb": row["write_mb"],
                    }
                )

            io_stats[phase] = stats
        else:
            pass

    return io_stats


def print_io_summary(io_stats: Dict[str, List[Dict[str, int]]]) -> None:
    """Print the summary of I/O information from io_stats loaded"""
    table_header = ["Device", "# of Read", "# of Write", "Read Size (MB)", "Write Size (MB)"]
    table = []

    for phase, stats in io_stats.items():
        for i, per_device_stats in enumerate(stats):
            row = [
                phase if i == 0 else "",
                per_device_stats["device"],
                per_device_stats["read_operations"],
                per_device_stats["read_mb"],
                per_device_stats["write_operations"],
                per_device_stats["write_mb"],
            ]
            table.append(row)

    print("\n[ I/O Statistic Summary ]")
    print(tabulate(table, table_header, tablefmt="fancy_grid"))
