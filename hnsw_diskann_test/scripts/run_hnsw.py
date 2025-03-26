from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import numpy as np
import time
import os

# 파라미터 설정
DIM = 128
TOTAL_VECTORS = 4_000_000
BATCH_SIZE = 100_000
NUM_QUERIES = 100
COLLECTION_NAME = "collection_hnsw_test_4M"
M = 30
efConstruction = 100
ef = 10
limit = 10
LOG_FILE = "4M_hnsw_4mem_2swapping_mmap.txt"

# 1. Milvus 연결
print("1. Milvus 서버에 연결 중...")
connections.connect(uri="http://localhost:19530", token="root:Milvus")
print("   연결 성공!\n")

# # 2. 기존 컬렉션 제거
# print(f"2. 기존 컬렉션 '{COLLECTION_NAME}' 삭제 시도...")
# if utility.has_collection(COLLECTION_NAME):
#     utility.drop_collection(COLLECTION_NAME)
#     print("   기존 컬렉션 삭제 완료!\n")
# else:
#     print("   삭제할 기존 컬렉션 없음\n")

# # 3. 컬렉션 스키마 정의 및 생성
# print("3. 컬렉션 및 스키마 생성 중...")
# fields = [
#     FieldSchema(name="my_id", dtype=DataType.INT64, is_primary=True),
#     FieldSchema(name="my_vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
# ]
# schema = CollectionSchema(fields)
# collection = Collection(name=COLLECTION_NAME, schema=schema)
# print("   컬렉션 생성 완료!\n")

# # 4. 데이터 삽입
# print("4. 데이터 삽입 중...")
# current_id = 0
# for batch_idx in range(TOTAL_VECTORS // BATCH_SIZE):
#     print(f"   Batch {batch_idx+1}/10 - 벡터 생성 및 삽입 중...")
#     vectors = np.random.random((BATCH_SIZE, DIM)).astype(np.float32)
#     ids = np.arange(current_id, current_id + BATCH_SIZE, dtype=np.int64)
#     collection.insert([ids, vectors])
#     current_id += BATCH_SIZE
# print("   전체 데이터 삽입 완료!\n")

# # 5. flush
# print("5. 디스크로 flush 중...")
# collection.flush()
# print("   flush 완료!\n")

# # 6. 인덱스 생성 및 빌드 시간 측정
# print("6. HNSW 인덱스 생성 및 빌드 대기 중...")
# start = time.perf_counter()
# collection.create_index(
#     field_name="my_vector",
#     index_name="hnsw_index",
#     index_params={
#         "index_type": "HNSW",
#         "metric_type": "L2",
#         "params": {"M": M, "efConstruction": efConstruction}
#     }
# )
# utility.wait_for_index_building_complete(COLLECTION_NAME, index_name="hnsw_index")
# end = time.perf_counter()
# print(f"   인덱스 빌드 완료! (소요 시간: {end - start:.3f}초)\n")

collection = Collection(name=COLLECTION_NAME)

# 7. 컬렉션 로드
print("7. 컬렉션 로드 중...")
collection.load()
print("   컬렉션 로드 완료!\n")

# 8. 검색 쿼리 수행 및 latency 측정
print(f"8. Top-{limit} 검색을 {NUM_QUERIES}회 반복하며 latency 측정 중...")
query_vectors = np.random.random((NUM_QUERIES, DIM)).astype(np.float32)
search_params = {"ef": ef}

with open(LOG_FILE, "w") as f:
    for i in range(NUM_QUERIES):
        start = time.perf_counter()
        results = collection.search(
            data=[query_vectors[i]],
            anns_field="my_vector",
            param=search_params,
            limit=limit,
            output_fields=["my_id"]
        )
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        f.write(f"{latency_ms:.3f} ms\n")
        print(f"   {i+1:3}/{NUM_QUERIES} - latency: {latency_ms:.3f} ms")

# 9. 로그 파일 경로 출력
print(f"\n9. 검색 latency 로그 저장 완료 → {os.path.abspath(LOG_FILE)}")
