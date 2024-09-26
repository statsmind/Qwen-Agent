from abc import ABC
from typing import List, Iterator, Union, Dict, Literal, Optional

from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message


class CachedModel(BaseChatModel, ABC):
    def __init__(self, origin_model: BaseChatModel):
        super().__init__()
        self.origin_model = origin_model

    def chat(
        self,
        messages: List[Union[Message, Dict]],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        extra_generate_cfg: Optional[Dict] = None,
    ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        self.origin_model.cache
        response = super().chat(messages, functions, stream, delta_stream, extra_generate_cfg)
        return response
