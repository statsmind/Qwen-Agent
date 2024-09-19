from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from qwen_agent.tools.base import load_tool


def test_pubmed_searcher():

    assistant = Assistant(
        function_list=[
            load_tool('web_searcher'),
            load_tool('pubmed_searcher')
        ]
    )

    *_, response = assistant.run([Message('user', 'Angong Niuhuang Pill有动物或临床研究的证据吗？')])
    print(response)