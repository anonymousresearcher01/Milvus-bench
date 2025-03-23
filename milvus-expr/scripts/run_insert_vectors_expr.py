import argparse
import os
import time
from typing import List
import numpy as np
import pandas as pd
import subprocess
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from io_utility import load_io_stats, print_io_summary
from plot_utility import plot_inser_vectors


def prepare_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="Text embeddings collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection: {collection_name}.")
    return collection


def insert_vectors(collection, vectors, texts, batch_size=1000) -> List[int]:
    total = vectors.shape[0]
    insert_times = []

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_vectors = vectors[i:end].tolist()
        batch_texts = texts[i:end].tolist()

        entities = [batch_vectors, batch_texts]

        batch_start = time.time()
        collection.insert(entities)
        batch_time = time.time() - batch_start
        insert_times.append(batch_time)

        print(f"Insert 진행률: {end}/{total}, 배치 시간: {batch_time:.4f}초")

    return insert_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Insertion Experiment")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples to use for insertion")
    parser.add_argument(
        "--data_path", type=str, default="/mnt/sda/milvus-io-test/data/random_generated/", help="Input data path"
    )
    args = parser.parse_args()

    print("Cleaning system cache...")
    try:
        subprocess.run(["sync"])
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        print("Cleaning system cache done.")
    except Exception as e:
        raise AssertionError(e)

    collection_name = f"test_collection_{args.num}"
    experiment_name = "insert_vectors"
    json_output_name = f"{experiment_name}_results_{args.num}.json"
    embeddings_file = os.path.join(args.data_path, f"text_embeddings_{args.num}.npy")
    metadata_file = os.path.join(args.data_path, f"text_metadata_{args.num}.csv")
    timing_stats = {
        "load_data": 0,
        "prepare_collection": 0,
        "insert_batches": [],
        "flush_to_disk": 0,
        "sync_disk": 0,
        "total": 0,
    }

    os.makedirs("../result_stat/io_monitoring", exist_ok=True)

    # 0. Preprocess
    print("Connecting to Milvus...")
    connections.connect("default", host="localhost", port="19530")

    if utility.has_collection(collection_name):
        print("Found existing collection. This is automatically supposed to be deleted now.")
        utility.drop_collection(collection_name)
        print(f"{collection_name} Deleted!")

    # Measure start time
    total_start = time.time()

    # 1. Load for embeddings and metadata files
    print("Loading embedding and metadata files...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "load_data"])

    load_start = time.time()
    embeddings = np.load(embeddings_file)
    metadata = pd.read_csv(metadata_file)
    dim = embeddings.shape[1]
    timing_stats["load_data"] = time.time() - load_start

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "load_data"])

    # 2. Prepare collection
    print("Preparing collection...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "prepare_collection"])

    prepare_start = time.time()
    collection = prepare_collection(collection_name, dim)
    timing_stats["prepare_collection"] = time.time() - prepare_start

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "prepare_collection"])

    # 3. Insert vector as a batch
    print("Inserting vectors...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "insert_vectors"])

    timing_stats["insert_batches"] = insert_vectors(collection, embeddings, metadata["text"])

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "insert_vectors"])

    # 4. Flush collection
    print("Flushing collection...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "flush_collection"])

    flush_start = time.time()
    collection.flush()  # explicit flush
    timing_stats["flush_to_disk"] = time.time() - flush_start

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "flush_collection"])

    # 5. Perform sync
    print("Synchronizing to the disk...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "sync_disk"])

    sync_start = time.time()
    subprocess.run(["sync"])
    timing_stats["sync_disk"] = time.time() - sync_start

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "sync_disk"])

    # Measure end time
    timing_stats["total"] = time.time() - total_start

    print("\n[ Expr.1 삽입 실험 결과 ]")

    print(f"총 실행 시간: {timing_stats['total']:.4f}초")
    print(f"데이터 로드: {timing_stats['load_data']:.4f}초 ({timing_stats['load_data']/timing_stats['total']*100:.2f}%)")
    print(
        f"컬렉션 준비: {timing_stats['prepare_collection']:.4f}초 ({timing_stats['prepare_collection']/timing_stats['total']*100:.2f}%)"
    )

    total_insert_time = sum(timing_stats["insert_batches"])
    print(f"벡터 삽입 총시간: {total_insert_time:.4f}초 ({total_insert_time/timing_stats['total']*100:.2f}%)")
    print(f"평균 배치 삽입 시간: {np.mean(timing_stats['insert_batches']):.4f}초")
    print(f"최대 배치 삽입 시간: {np.max(timing_stats['insert_batches']):.4f}초")
    print(f"최소 배치 삽입 시간: {np.min(timing_stats['insert_batches']):.4f}초")
    print(f"배치 삽입 시간 표준편차: {np.std(timing_stats['insert_batches']):.4f}초")

    vectors_per_second = len(embeddings) / total_insert_time
    print(f"초당 삽입 벡터 수: {vectors_per_second:.2f} vectors/s")

    print(
        f"컬렉션 flush: {timing_stats['flush_to_disk']:.4f}초 ({timing_stats['flush_to_disk']/timing_stats['total']*100:.2f}%)"
    )
    print(f"sync 명령: {timing_stats['sync_disk']:.4f}초 ({timing_stats['sync_disk']/timing_stats['total']*100:.2f}%)")

    io_stats = load_io_stats(experiment_name)
    print_io_summary(io_stats)

    with open(f"../result_stat/{json_output_name}", "w") as f:
        combined_stats = {"timing": timing_stats, "io_stats": io_stats}
        json.dump(combined_stats, f, indent=2)

    plot_inser_vectors(experiment_name, timing_stats, args.num, io_stats)
    print(f"\n Complete Expr1. The result has been stored to {json_output_name}")
