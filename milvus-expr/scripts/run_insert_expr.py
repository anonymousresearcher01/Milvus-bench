import argparse
import os
import time
import numpy as np
import pandas as pd
import subprocess
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


def prepare_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        print("Found existing collection. This is automatically supposed to be deleted now.")
        utility.drop_collection(collection_name)
        print(f"'{collection_name}' Deleted!")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="Text embeddings collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection: '{collection_name}'.")
    return collection


def insert_vectors(collection, vectors, texts, batch_size=1000):
    total = vectors.shape[0]
    insert_times = []

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batch_vectors = vectors[i:end].tolist()
        batch_texts = texts[i:end].tolist()

        entities = [batch_vectors, batch_texts]

        start_batch = time.time()
        collection.insert(entities)
        end_batch = time.time()
        batch_time = end_batch - start_batch
        insert_times.append(batch_time)

        print(f"Insert 진행률: {end}/{total}, 배치 시간: {batch_time:.4f}초")

    return insert_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Insertion Expr")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples to use for insertion")
    parser.add_argument(
        "--data_path", type=str, default="/mnt/sda/milvus-io-test/data/random_generated/", help="Input data path"
    )
    args = parser.parse_args()

    collection_name = f"test_collection_{args.num}"
    embeddings_file = os.path.join(args.data_path, f"text_embeddings_{args.num}.npy")
    metadata_file = os.path.join(args.data_path, f"text_metadata_{args.num}.csv")

    subprocess.run(["bash", "-c", "source ./io_monitor.sh; start_monitoring insert"])

    print("Connecting to Milvus...")
    connections.connect("default", host="localhost", port="19530")

    print(f"Embedding Data Loading: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"Metadata Loading: {metadata_file}")
    metadata = pd.read_csv(metadata_file)

    dim = embeddings.shape[1]
    print(f"Check embedding dim: {dim}")
    print(f"Check total # of vectors: {embeddings.shape[0]}")

    start_time = time.time()
    print("Preparing collection...")
    collection = prepare_collection(collection_name, dim)

    print("Inserting vectors...")
    insert_times = insert_vectors(collection, embeddings, metadata["text"])

    print("Data flushing...")
    flush_start = time.time()
    collection.flush()  # explicit flush
    flush_end = time.time()

    end_time = time.time()

    # NOTE(dhmin): Other metric will be introduced.
    total_time = end_time - start_time
    flush_time = flush_end - flush_start
    avg_batch_time = sum(insert_times) / len(insert_times)
    avg_vectors_per_sec = embeddings.shape[0] / total_time

    print("\n===== Expr.1 삽입 실험 결과 =====")
    print(f"총 벡터 수: {embeddings.shape[0]}")
    print(f"총 삽입 시간: {total_time:.2f}초")
    print(f"Flush 시간: {flush_time:.2f}초")
    print(f"평균 배치 삽입 시간: {avg_batch_time:.4f}초")
    print(f"벡터/초: {avg_vectors_per_sec:.2f}")

    results = {
        "total_vectors": embeddings.shape[0],
        "dimension": dim,
        "total_time(s)": total_time,
        "flush_time(s)": flush_time,
        "avg_batch_time(s)": avg_batch_time,
        "vectors_per_second(s)": avg_vectors_per_sec,
        "batch_times(s)": insert_times,
    }

    import json

    with open(f"../result_stat/insert_results_{args.num}.json", "w") as f:
        json.dump(results, f, indent=2)

    subprocess.run(["bash", "-c", "source ./io_monitor.sh; stop_monitoring insert"])
    print(f"\n Complete Expr1. The result has been stored to insert_results_{args.num}.json")
