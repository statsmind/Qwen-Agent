from typing import Iterator, List, Optional, Union, Dict

from qwen_agent.agents.assistant import Assistant
from qwen_agent.agents.write_from_scratch import WriteFromScratch
from qwen_agent.agents.writing import ContinueWriting
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, CONTENT, Message, DEFAULT_SYSTEM_MESSAGE
from qwen_agent.tools import BaseTool


class ArticleAgent(Assistant):
    """This is an agent for writing articles.

    It can write a thematic essay or continue writing an article based on reference materials
    """
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = 'Assistant',
                 description: Optional[str] = 'Assistant',
                 files: Optional[List[str]] = None,
                 rag_cfg: Optional[Dict] = None,
                 knowledge: Optional[str] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         rag_cfg=rag_cfg,
                         **kwargs)
        self.knowledge = knowledge

    def _run(self,
             messages: List[Message],
             lang: str = 'en',
             full_article: bool = True,
             **kwargs) -> Iterator[List[Message]]:

        if not self.knowledge:
            # Need to use Memory agent for data management
            *_, last = self.mem.run(messages=messages, **kwargs)
            _ref = last[-1][CONTENT]
        else:
            _ref = self.knowledge

        response = []
        if _ref:
            response.append(Message(ASSISTANT, f'>\n> Search for relevant information: \n{_ref}\n'))
            yield response

        if full_article:
            writing_agent = WriteFromScratch(llm=self.llm)
        else:
            writing_agent = ContinueWriting(llm=self.llm)
            response.append(Message(ASSISTANT, '>\n> Writing Text: \n'))
            yield response

        for rsp in writing_agent.run(messages=messages, lang=lang, knowledge=_ref):
            if rsp:
                yield response + rsp
