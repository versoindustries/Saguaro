import time
import hashlib
import numpy as np


def encode_text_mock(text, dim):
    max_len = 512
    text = text[: max_len * 10]
    chunks = [text[i : i + 4] for i in range(0, len(text), 4)]
    chunks = chunks[:max_len]

    vecs = []
    for chunk in chunks:
        h = hashlib.md5(chunk.encode()).hexdigest()
        seed = int(h, 16) % (2**32)
        rng = np.random.default_rng(seed)
        vecs.append(rng.normal(0, 0.1, size=(dim,)))
    return np.array(vecs)


dim = 8192
text = "a" * 10000

start = time.time()
print(f"Benchmarking encoding for 10k chars at dim {dim}...")
encode_text_mock(text, dim)
end = time.time()
print(f"Time for 10k chars: {end - start:.4f}s")
