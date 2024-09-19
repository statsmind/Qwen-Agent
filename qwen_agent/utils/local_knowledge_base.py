from typing import List

LOCAL_KB_CACHE = {}


class LocalKnowledgeBase(object):
    def add_knowledge(self, record: List[dict]) -> str:
        kb_id = f"kb:{len(LOCAL_KB_CACHE)}"
        LOCAL_KB_CACHE[kb_id] = record
        return kb_id

    def get_knowledge(self, id: str) -> List[dict]:
        return LOCAL_KB_CACHE[id]

