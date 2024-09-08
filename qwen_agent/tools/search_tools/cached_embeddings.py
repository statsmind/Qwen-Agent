import hashlib
import os
from typing import List, Literal

import numpy as np
from langchain_core.embeddings import Embeddings
from qwen_agent.log import logger


class CachedEmbeddings(Embeddings):
    def __init__(self, embeddings: Embeddings, cache_folder: str = "/tmp/embeddings"):
        self.embeddings = embeddings
        self.cache_folder = cache_folder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text, 'document') for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, 'query')

    def _embed(self, text: str, type: Literal['document', 'query']) -> List[float]:
        hash = self.md5(text)
        cache_path = os.path.join(
            self.cache_folder, str(self.embeddings.__class__.__name__), type,
            hash[0], hash[1])
        cache_file = os.path.join(cache_path, hash + ".npy")
        logger.info("cache file: {}".format(cache_file))
        os.makedirs(cache_path, exist_ok=True)

        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            return np.load(cache_file)

        if type == 'document':
            vector = self.embeddings.embed_documents([text])[0]
        else:
            vector = self.embeddings.embed_query(text)

        np.save(cache_file, vector)
        return vector

    def md5(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
