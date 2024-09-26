from typing import List, Literal, Iterator

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message


class PMCAgent(Assistant):
    """
    PMC full text engine
    """
    def _run(self,
             messages: List[Message],
             lang: Literal['en', 'zh'] = 'en',
             knowledge: str = '',
             **kwargs) -> Iterator[List[Message]]:
        pass
