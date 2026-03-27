import argparse
import time
import json
import subprocess
from pymilvus import connections, Collection

from io_utility import load_io_stats, print_io_summary
from plot_utility import plot_build_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Build Experiment")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()

    print("Cleaning system cache...")
    try:
        subprocess.run(["sync"])
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        print("Cleaning system cache done.")
    except Exception as e:
        raise AssertionError(e)

    collection_name = f"test_collection_{args.num}"
    experiment_name = "build_index"
    json_output_name = f"{experiment_name}_results_{args.num}.json"
    # TODO(Dhmin): Other index type would be supported via argparser, but supports HNSW only now
    index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
    timing_stats = {"build_index": 0, "sync_disk": 0, "total": 0}

    # sanity check
    # if not utility.has_collection(collection_name):
    #     raise AssertionError(f"The {collection_name} collection does not exists.")

    # 0. Preprocess
    print(f"Connecting to Milvus and loading the collection {collection_name}...")
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(name=collection_name)
    print(f"Check collection info: {collection.num_entities}, Schema: {collection.schema}")

    if collection.has_index():
        print("Found existing index. This is now being deleted automatically.")
        collection.drop_index()
        print("Index Deleted!")

    # Measure start time
    total_start = time.time()

    # 1. Create index
    print("Building index...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "build_index"])

    build_index_start = time.time()
    collection.create_index("vector", index_params)
    timing_stats["build_index"] = time.time() - build_index_start

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "build_index"])

    # 2. Perform sync (by using collection.release() API)
    print("Synchronizing to the disk...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "sync_disk"])

    sync_start = time.time()
    collection.release()
    subprocess.run(["sync"])
    timing_stats["sync_disk"] = time.time() - sync_start

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "sync_disk"])

    timing_stats["total"] = time.time() - total_start

    print("\n[ Expr.2 인덱스 빌드 실험 결과 ]")
    print(f"총 인덱스 빌드 소요 시간: {timing_stats['total']:.2f}초")

    index_info = collection.index().params
    print(f"생성된 인덱스 정보: {index_info}")

    io_stats = load_io_stats(experiment_name)
    print_io_summary(io_stats)

    with open(f"../result_stat/{json_output_name}", "w") as f:
        combined_stats = {
            "collection_name": collection_name,
            "num_vectors": collection.num_entities,
            "index_type": "HNSW",
            "index_params": index_params,
            "timing": timing_stats,
            "io_stats": io_stats,
        }
        json.dump(combined_stats, f, indent=2)

    plot_build_index(experiment_name, timing_stats, args.num, io_stats)
    print(f"\nComplete Expr2. The result has been stored to {json_output_name}")
