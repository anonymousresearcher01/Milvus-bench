
# Milvus (standalone)

## Installing Milvus on Ubuntu (docker basis)
Follow the below to install docker and docker-compose.

```bash
sudo apt-get update
sudo apt-get upgrade -y

# Docker 설치에 필요한 패키지 설치
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Docker 공식 GPG 키 추가
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Docker 저장소 추가
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Docker 설치
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Docker 서비스 시작 및 자동 시작 설정
sudo systemctl start docker
sudo systemctl enable docker

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 docker 명령어 사용 가능)
sudo usermod -aG docker $USER
```

To install Milvus (Standalone), use

```bash
mkdir -p ~/milvus-io-test
cd ~/milvus-io-test

# docker-compose.yml 파일 생성
cat > docker-compose.yml << EOF
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${PWD}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${PWD}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${PWD}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
EOF

# 필요한 볼륨 디렉토리 생성
mkdir -p volumes/etcd volumes/minio volumes/milvus

# Milvus 및 의존성 컨테이너 시작
docker-compose up -d
```

## Error reports
When `drop_collection()` is being executed, you may face failed error of `InvalidateCollectionMetaCache` since the inconsistency between Node IDs of Proxy instances. It is desirable to run below, and then re-run the docker composition.

```bash
rm -rf volumes/etcd volumes/milvus volumes/minio
```

Or give it a try to implement invalidation before dropping the collection in the python code.

```python
collection_name = "my_collection_name"

# Enforcing cache invalidation
utility.invalidate_collection_cache(collection_name)

utility.drop_collection(collection_name)
```


## Supported scripts
The supported scripts are stored in `scripts` directory.
- `docker_monitor.py`: docker stat monitoring script
- `generate_random_vector.py`: script for generating random texts with embedding results
  - The output directory must be adjusted by the user.
- `run_insert_expr.py`: script for inserting embedding vectors (The L stage of ETL pipeline)
- `run_index_build_expr.py`: script for building index using stored vectors
- `run_index_load_expr.py`: script for loading the index from the disk to the memory.
- `run_search_expr.py`: script for searching top-k using index and original texts.
