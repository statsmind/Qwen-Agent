import copy
from typing import List, Literal, Iterator

from qwen_agent import Agent
from qwen_agent.llm.schema import Message, CONTENT


class GenQuery(Agent):
    PROMPT_TEMPLATE_ZH = """{user_question}
-------
鉴于我上面的问题，您的任务是将我的问题改写成一个语义完整且清晰明了的独立问题。
如果我的问题在语义上已经完整或与之前的聊天记录无关，请返回我的问题作为独立问题。
如果我没有指定任何语言，则默认使用中文来回答独立问题。
-------
独立问题："""

    PROMPT_TEMPLATE_EN = """{user_question}
-------
Given my above question, your task is to rephrase my question into a standalone question which is semantically complete and articulated.
If my question is semantically complete already or not related to chat history, return the question as is for the standalone question.
If I do not specify any language, then default to using Chinese for the standalone question.
-------
Standalone Question:"""

    PROMPT_TEMPLATE = {
        'zh': PROMPT_TEMPLATE_ZH,
        'en': PROMPT_TEMPLATE_EN,
    }

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'zh', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        messages[-1][CONTENT] = self.PROMPT_TEMPLATE[lang].format(user_question=messages[-1].content)
        return self._call_llm(messages=messages)
