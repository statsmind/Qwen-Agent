import json
import os
from typing import List, Tuple

import torch

from qwen_agent.tools.base import register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import BaseSearch
from qwen_agent.tools.search_tools.cached_embeddings import CachedEmbeddings


@register_tool('vector_search')
class VectorSearch(BaseSearch):
    # TODO: Optimize the accuracy of the embedding retriever.

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        # TODO: More types of embedding can be configured
        try:
            from langchain.schema import Document
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please install langchain by: `pip install langchain`')
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            from langchain_community.embeddings import OllamaEmbeddings
            from langchain_community.vectorstores import FAISS
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please install langchain_community by: `pip install langchain_community`, '
                'and install faiss by: `pip install faiss-cpu` or `pip install faiss-gpu` (for CUDA supported GPU)')
        # Extract raw query
        try:
            query_json = json.loads(query)
            # This assumes that the user's input will not contain json str with the 'text' attribute
            if 'text' in query_json:
                query = query_json['text']
        except json.decoder.JSONDecodeError:
            pass

        # Plain all chunks from all docs
        all_chunks = []
        for doc in docs:
            for chk in doc.raw:
                all_chunks.append(Document(page_content=chk.content[:2000], metadata=chk.metadata))

        # embeddings = DashScopeEmbeddings(model='text-embedding-v2',
        #                                  dashscope_api_key=os.getenv('DASHSCOPE_API_KEY', ''))
        bge_embeddings = HuggingFaceBgeEmbeddings(
            model_name='BAAI/bge-m3',
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={'normalize_embeddings': True}
        )
        embeddings = CachedEmbeddings(bge_embeddings)
        db = FAISS.from_documents(all_chunks, embeddings)
        chunk_and_score = db.similarity_search_with_score(query, k=len(all_chunks))

        return [(chk.metadata['source'], chk.metadata['chunk_id'], 1.0 / (score + 1e-6)) for chk, score in
                chunk_and_score]
