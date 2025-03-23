#!/usr/bin/env python3
# run_search_experiment.py
import os
import argparse
import time
import numpy as np
import json
import subprocess
import random
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility

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


def run_search(query_texts, top_k=10):
    """Run vectorDB search"""
    search_latencies = []
    all_results = []

    print(f"Searching {len(query_texts)} queries...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vectors = model.encode(query_texts)

    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    for i, query_vector in enumerate(query_vectors):
        print(f"Query {i+1}/{len(query_texts)} Execution: '{query_texts[i]}'")

        start_time = time.time()
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["text"],
        )
        end_time = time.time()

        latency = end_time - start_time
        search_latencies.append(latency)

        query_results = []
        for hits in results:
            for hit in hits:
                query_results.append(
                    {
                        "id": hit.id,
                        "distance": hit.distance,
                        "text": (hit.entity.get("text") or "")[:100] + "..."
                        if len(hit.entity.get("text") or "") > 100
                        else hit.entity.get("text") or "",
                    }
                )

        all_results.append({"query": query_texts[i], "latency": latency, "results": query_results})

        print(f"  - 검색 완료. 지연 시간: {latency:.4f}초")

    return search_latencies, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Build Expr")
    parser.add_argument("--num", type=int, default=1000, help="Number of samples")
    parser.add_argument("--query", type=int, default=50, help="Number of queries to test")
    parser.add_argument("--topk", type=int, default=10, help="Top-k")
    args = parser.parse_args()
    collection_name = f"test_collection_{args.num}"
    num_queries = args.query
    top_k = args.topk

    # subprocess.run(["sudo", "bash", "./io_monitor.sh", "start_monitoring", "search"])
    # time.sleep(5)

    print("Connecting to Milvus...")
    connections.connect("default", host="localhost", port="19530")
    print(f"Collection Loading: '{collection_name}'...")
    # existing_collection = utility.list_collections()

    collection = Collection(name=collection_name)

    if collection.has_index():
        collection.load()
        print("Collected Loaded!")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"Generating random queries of {num_queries}...")
    query_texts = [generate_random_query() for _ in range(num_queries)]

    print("Executing warm-up query...")
    warmup_query = "my decision is to dive everyday to the history without thinking"
    warmup_vector = model.encode([warmup_query])[0]
    collection.search(
        data=[warmup_vector.tolist()],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=5,
    )

    print("\nSearching actual query...")
    start_time = time.time()
    search_latencies, all_results = run_search(query_texts, top_k)
    end_time = time.time()

    # 결과 분석
    total_time = end_time - start_time
    avg_latency = sum(search_latencies) / len(search_latencies)
    min_latency = min(search_latencies)
    max_latency = max(search_latencies)
    p95_latency = sorted(search_latencies)[int(len(search_latencies) * 0.95)]
    qps = num_queries / total_time

    print("\n===== Expr.4 검색 실험 결과 =====")
    print(f"총 쿼리 수: {num_queries}")
    print(f"총 실행 시간: {total_time:.2f}초")
    print(f"평균 지연 시간: {avg_latency:.4f}초")
    print(f"최소 지연 시간: {min_latency:.4f}초")
    print(f"최대 지연 시간: {max_latency:.4f}초")
    print(f"95퍼센타일 지연 시간: {p95_latency:.4f}초")
    print(f"초당 쿼리 수(QPS): {qps:.2f}")

    results = {
        "total_queries": num_queries,
        "total_time": total_time,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "p95_latency": p95_latency,
        "qps": qps,
        "individual_latencies": search_latencies,
        "query_results": all_results,
    }

    with open(f"../result_stat/search_results_{args.num}.json", "w") as f:
        json.dump(results, f, indent=2)

    # time.sleep(5)
    # subprocess.run(["sudo", "bash", "./io_monitor.sh", "stop_monitoring search"])
    print(f"\n Complete Expr4. The result has been stored to search_results_{args.num}.json")
