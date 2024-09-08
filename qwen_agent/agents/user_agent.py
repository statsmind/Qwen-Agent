from typing import Iterator, List, Optional, Union, Dict

from qwen_agent.agent import Agent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message, DEFAULT_SYSTEM_MESSAGE
from qwen_agent.tools import BaseTool

PENDING_USER_INPUT = '<!-- INTERRUPT: PENDING_USER_INPUT -->'


class UserAgent(Agent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 should_interrupt: bool = False,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description)

        self.should_interrupt = should_interrupt

    def _run(self, messages: List[Message], **kwargs) -> Iterator[List[Message]]:
        yield [Message(role='user', content=PENDING_USER_INPUT, name=self.name)]
