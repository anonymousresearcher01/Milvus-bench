import os
import argparse
import sqlite3
import time
import json
import subprocess
import random
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

from io_utility import load_io_stats, print_io_summary
from plot_utility import plot_search_vectors

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_random_query(num_words=5):
    """Generate random query using some bunch of english words"""
    words = [
        "data",
        "vector",
        "search",
        "database",
        "embedding",
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "algorithm",
        "neural",
        "network",
        "model",
        "query",
        "result",
        "similarity",
        "distance",
        "index",
        "storage",
        "efficiency",
        "kaftan",
        "spartan",
        "suntan",
        "tan",
        "stable",
        "loss",
        "improve",
        "step",
        "partner",
        "research",
        "developer",
        "english",
        "organization",
        "silver",
        "deligent",
        "linging",
        "awkward",
        "amenity",
        "mine",
        "stock",
        "right",
        "consequence",
        "traveller",
        "firm",
        "station",
        "sheet",
        "pressure",
        "releatively",
        "wrap",
        "again",
        "against",
        "age",
        "agency",
        "agent",
        "ago",
        "change",
        "character",
        "charge",
    ]

    return " ".join(random.sample(words, k=min(num_words, len(words))))


def run_search(db_file, query_texts, query_vectors, top_k=10):
    """Run vectorDB search"""
    search_latencies = []
    all_results = []

    print(f"Searching {len(query_vectors)} queries...")
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    for i, query_vector in enumerate(query_vectors):
        print(f"Query {i+1}/{len(query_vectors)} Execution: '{query_texts[i]}'")

        start_time = time.time()
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["embedding_index"],
        )

        query_results = []
        for hits in results:
            for hit in hits:
                cursor.execute("SELECT text FROM embeddings WHERE embedding_index=?", (hit.embedding_index,))
                row = cursor.fetchone()
                original_text = row[0] if row else "Not found"

                query_results.append(
                    {
                        "id": hit.id,
                        "distance": hit.distance,
                        "embedding_index": hit.embedding_index,
                        "original_text": original_text,
                        # "text": (hit.entity.get("text") or "")[:100] + "..."
                        # if len(hit.entity.get("text") or "") > 100
                        # else hit.entity.get("text") or "",
                    }
                )

        latency = time.time() - start_time
        search_latencies.append(latency)

        all_results.append({"query": query_texts[i], "latency": latency, "results": query_results})
        print(f"  - Done to search. elapsed time: {latency:.4f} sec")

    conn.close()

    return search_latencies, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Build Expr")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--data_path", type=str, default="/mnt/sda/milvus-io-test/data/random_generated/", help="Input data path"
    )
    parser.add_argument("--query", type=int, default=100, help="Number of queries to test")
    parser.add_argument("--topk", type=int, default=10, help="Top-k")
    args = parser.parse_args()
    collection_name = f"test_collection_{args.num}"
    experiment_name = "search_vectors"
    json_output_name = f"{experiment_name}_results_{args.num}.json"
    metadata_file = os.path.join(args.data_path, f"text_metadata_{args.num}.csv")
    num_queries = args.query
    top_k = args.topk
    timing_stats = {"search_vectors": 0, "total": 0}

    # 0. Preprocess
    print("Connecting to Milvus and loading collection...")
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(name=collection_name)

    if collection.has_index():
        collection.load()
        print("Collected Loaded!")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Executing warm-up query...")
    warmup_query = "my decision is to dive everyday to the history without thinking"
    warmup_vector = model.encode([warmup_query])[0]
    collection.search(
        data=[warmup_vector.tolist()],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=5,
    )
    print(f"Generating random queries of {num_queries}...")
    query_texts = [generate_random_query() for _ in range(num_queries)]
    query_vectors = model.encode(query_texts)

    # create_db_from_file(metadata_file)
    db_file = "embedding_text.db"

    # Measure the start time
    total_start = time.time()

    # 1. Search vectors
    print("\nSearching actual query...")
    subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", experiment_name, "search_vectors"])

    search_vectors_start = time.time()
    search_latencies, all_results = run_search(db_file, query_texts, query_vectors, top_k)
    timing_stats["search_vectors"] = time.time() - search_vectors_start

    subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring", experiment_name, "search_vectors"])

    timing_stats["total"] = time.time() - total_start

    # 결과 분석
    avg_latency = sum(search_latencies) / len(search_latencies)
    min_latency = min(search_latencies)
    max_latency = max(search_latencies)
    p95_latency = sorted(search_latencies)[int(len(search_latencies) * 0.95)]
    qps = num_queries / timing_stats["total"]

    print("\n[ Expr.4 벡터 탐색 실험 결과 ]")
    print(f"총 쿼리 수: {num_queries}")
    print(f"총 실행 시간: {timing_stats['total']:.2f}초")
    print(f"평균 지연 시간: {avg_latency:.4f}초")
    print(f"최소 지연 시간: {min_latency:.4f}초")
    print(f"최대 지연 시간: {max_latency:.4f}초")
    print(f"95퍼센타일 지연 시간: {p95_latency:.4f}초")
    print(f"초당 쿼리 수(QPS): {qps:.2f}")

    io_stats = load_io_stats(experiment_name)
    print_io_summary(io_stats)

    with open(f"../result_stat/{json_output_name}", "w") as f:
        combined_stats = {
            "timing": timing_stats,
            "io_stats": io_stats,
            "total_queries": num_queries,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "p95_latency": p95_latency,
            "qps": qps,
            "individual_latencies": search_latencies,
            "query_results": all_results,
        }
        json.dump(combined_stats, f, indent=2)

    plot_search_vectors(experiment_name, timing_stats, args.num, io_stats)
    print(f"\nComplete Expr4. The result has been stored to {json_output_name}")
