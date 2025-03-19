import argparse
import gc
import numpy as np
import pandas as pd
import os
import random
import string
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_random_text(min_words=20, max_words=80):
    """Generate random str text"""
    num_words = random.randint(min_words, max_words)
    words = []
    for _ in range(num_words):
        word_len = random.randint(3, 12)
        word = "".join(random.choice(string.ascii_lowercase) for _ in range(word_len))
        words.append(word)
    return " ".join(words)


parser = argparse.ArgumentParser(description="Milvus NQ Performance Test")
parser.add_argument(
    "--out_path",
    "-o",
    type=str,
    default="/mnt/sda/milvus-io-test/data/random_generated/",
    help="Path for storing results",
)
parser.add_argument("--num", type=int, default=1000, help="Number of samples to use for generating")
args = parser.parse_args()

data_dir = args.out_path  # "/mnt/sda/milvus-io-test/data/random_generated/"
os.makedirs(data_dir, exist_ok=True)

# Config for generating virtual texts
num_texts = args.num  # number of texts
batch_size = 1000
num_batches = (num_texts + (batch_size - 1)) // batch_size

# Set file path
text_file = os.path.join(data_dir, f"random_texts_{args.num}.txt")
embeddings_file = os.path.join(data_dir, f"text_embeddings_{args.num}.npy")
metadata_file = os.path.join(data_dir, f"text_metadata_{args.num}.csv")

#NOTE(Dhmin): Not desirable 'Extract -> Embedding model load -> Transform -> File write' due to OOM
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = None

with open(text_file, "w") as f:
    pass

# Store initial metadata
metadata = pd.DataFrame(columns=["text", "embedding_index"])
metadata.to_csv(metadata_file, index=False)

def process_batch(start_idx: int, end_idx: int , current_batch_size: int, embedding_dim: int) -> int:
    batch_texts = [generate_random_text() for _ in range(current_batch_size)]

    # 1. Extract
    print(f"Generating {current_batch_size} random texts...")
    with open(text_file, "a") as f:
        for text in batch_texts:
            f.write(text + "\n")
    
    # 2. Transform
    print("Embedding batch...")
    batch_embeddings = model.encode(batch_texts)
    local_embedding_dim = batch_embeddings.shape[1]
    if embedding_dim is None:
        print(f"Embedding dimensions: {batch_embeddings.shape}")
        print(f"Embedding dim: {local_embedding_dim}")

        fp = np.memmap(embeddings_file, dtype=np.float32, mode="w+", shape=(num_texts, local_embedding_dim))
        fp.flush()
    else:
        assert local_embedding_dim == embedding_dim, "Found the different embedding dims!"
    
    fp = np.memmap(embeddings_file, dtype=np.float32, mode="r+", shape=(num_texts, local_embedding_dim))
    fp[start_idx:end_idx] = batch_embeddings.astype(np.float32)
    fp.flush()

    # 3. Metadata
    batch_metadata = pd.DataFrame({
        "text": batch_texts,
        "embedding_index": range(start_idx, end_idx)
    })
    batch_metadata.to_csv(metadata_file, index=False, mode="a", header=False)

    del batch_texts
    del batch_embeddings
    del batch_metadata
    gc.collect()
    print(f"Completed batch {batch_idx + 1}, memory cleaned.")
    return local_embedding_dim


print(f"Processing {num_texts} texts in {num_batches} batches...")
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_texts)
    current_batch_size = end_idx - start_idx
    
    print(f"\nBatch {batch_idx + 1}/{num_batches}: Processing items {start_idx} to {end_idx-1}")
    batch_embedding_dim = process_batch(start_idx, end_idx, current_batch_size, embedding_dim)
    if embedding_dim is None:
        embedding_dim = batch_embedding_dim

# 1. Extract
# print(f"Creating: {num_texts} random texts...")
# texts = []
# for i in tqdm(range(num_texts)):
#     texts.append(generate_random_text())


# with open(text_file, "w") as f:
#     for text in texts:
#         f.write(text + "\n")
# print(f"Completed to store text as a file: {text_file}")

# 2. Transform
# print("Embedding...")

# embeddings = []
# for i in tqdm(range(0, len(texts), batch_size)):
#     batch_texts = texts[i : i + batch_size]
#     batch_embeddings = model.encode(batch_texts)
#     embeddings.extend(batch_embeddings)

#     del batch_texts
#     del batch_embeddings
#     gc.collect()


# embeddings_array = np.array(embeddings).astype(np.float32)
# np.save(embeddings_file, embeddings_array)

print("Completed:")
print(f"- Original texts: {text_file}")
print(f"- Embedding data: {embeddings_file}")
print(f"- Metadata: {metadata_file}")
print(f"Embedding Dimension: {embedding_dim}")
print(f"Total # of data: {num_texts}")
