import argparse
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

# 1. Extract
print(f"Creating: {num_texts} random texts...")
texts = []
for i in tqdm(range(num_texts)):
    texts.append(generate_random_text())

text_file = os.path.join(data_dir, f"random_texts_{args.num}.txt")
with open(text_file, "w") as f:
    for text in texts:
        f.write(text + "\n")
print(f"Completed to store text as a file: {text_file}")

# 2. Transform
print("Embedding...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i : i + batch_size]
    batch_embeddings = model.encode(batch_texts)
    embeddings.extend(batch_embeddings)

embeddings_array = np.array(embeddings).astype(np.float32)
embeddings_file = os.path.join(data_dir, f"text_embeddings_{args.num}.npy")
np.save(embeddings_file, embeddings_array)

# Store metadata for connecting text and embedding data
metadata = pd.DataFrame({"text": texts, "embedding_index": range(len(texts))})
metadata_file = os.path.join(data_dir, f"text_metadata_{args.num}.csv")
metadata.to_csv(metadata_file, index=False)

print("Completed:")
print(f"- Original texts: {text_file}")
print(f"- Embedding data: {embeddings_file}")
print(f"- Metadata: {metadata_file}")
print(f"Embedding Dimension: {embeddings_array.shape[1]}")
print(f"Total # of data: {len(texts)}")
