from abc import ABC
from typing import List, Union

from qwen_agent.utils.tokenization_qwen import count_tokens

LOCAL_KB_CACHE = {}


class GlobalKnowledgeBase(object):
    def add_knowledge(self, record: Union[str, List[dict]]) -> str:
        if isinstance(record, str):
            record = [{"content": [{"text": record}]}]

        kb_index = 1 + len(LOCAL_KB_CACHE)
        kb_id = f"kb:{kb_index}"

        for index, doc in enumerate(record):
            if isinstance(doc, str):
                doc = {"content": [{"text": record}]}

            if 'page_num' not in doc:
                doc['page_num'] = index + 1

            if 'title' not in doc:
                doc['title'] = f"[^{kb_index}]"

            for c in doc['content']:
                if 'token' not in c:
                    c['token'] = count_tokens(c['text'])

            record[index] = doc

        LOCAL_KB_CACHE[kb_id] = record
        return kb_id

    def get_knowledge(self, id: str) -> List[dict]:
        return LOCAL_KB_CACHE[id]

