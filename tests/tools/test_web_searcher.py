from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from qwen_agent.tools.base import load_tool


def test_web_searcher():

    assistant = Assistant(
        function_list=[
            load_tool('web_searcher')
        ]
    )

    *_, response = assistant.run([Message('user', '安宫牛黄丸英文怎么表达？')])
    print(response)