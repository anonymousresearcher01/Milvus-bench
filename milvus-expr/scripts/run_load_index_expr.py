import argparse
import time
import json
import subprocess
import psutil
from pymilvus import connections, Collection

from io_utility import load_io_stats, print_io_summary
from plot_utility import plot_load_index


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB unit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Load Experiment")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()

    collection_name = f"test_collection_{args.num}"
    experiment_name = "load_index"
    json_output_name = f"{experiment_name}_results_{args.num}.json"
    timing_stats = {"load_index": 0, "total": 0}

    # NOTE(Dhmin): Release pre-loaded index
    print("Releasing pre-loaded collection...")
    try:
        Collection(collection_name).release()
    except Exception as e:
        print(e)

    # NOTE(Dhmin): Clean the system cache
    print("Cleaning system cache...")
    try:
        subprocess.run(["sync"])
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        print("Cleaning system cache done.")
    except Exception as e:
        raise AssertionError(e)

    # 0. Preprocess
    print(f"Connecting to Milvus and loading the collection {collection_name}...")
    connections.connect("default", host="localhost", port="19530")

    before_load_memory = get_memory_usage()
    print(f"Before Memory Usage: {before_load_memory:.2f} MB")

    collection = Collection(name=collection_name)
    print(f"Check collection info: {collection.num_entities}, Schema: {collection.schema}")

    # Measure the start time
    total_start = time.time()

    # 1. Load Collection
    print("Loading index...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "load_index"])

    load_start_time = time.time()
    collection.load()
    timing_stats["load_index"] = time.time() - load_start_time
    end_time = time.time()

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "load_index"])

    timing_stats["total"] = time.time() - total_start

    after_load_memory = get_memory_usage()
    memory_increase = after_load_memory - before_load_memory

    print("\n[ Expr.3 인덱스 로드 실험 결과 ]")
    print(f"총 인덱스 로드 완료 소요 시간: {timing_stats['total']:.2f}초")
    print(f"After Memory Usage: {after_load_memory:.2f} MB (증가량: {memory_increase:.2f} MB)")

    try:
        index_info = collection.index().params
        print(f"로드된 인덱스 정보: {index_info}")
    except Exception as e:
        print(f"Fail to retrieve index_info: {str(e)}")

    io_stats = load_io_stats(experiment_name)
    print_io_summary(io_stats)

    with open(f"../result_stat/{json_output_name}", "w") as f:
        combined_stats = {
            "collection_name": collection_name,
            "memory_before_load_mb": before_load_memory,
            "memory_after_load_mb": after_load_memory,
            "memory_increase_mb": memory_increase,
            "timing": timing_stats,
            "io_stats": io_stats,
        }
        json.dump(combined_stats, f, indent=2)

    plot_load_index(experiment_name, timing_stats, args.num, io_stats)
    print(f"\nComplete Expr3. The result has been stored to {json_output_name}")
