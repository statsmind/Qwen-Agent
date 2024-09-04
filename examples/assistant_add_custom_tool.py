"""An image generation agent implemented by assistant"""

import json
import os
import urllib.parse

import json5

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


# Add a custom tool named my_image_gen：
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI painting (image generation) service, input text description, and return the image URL drawn based on text information.'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Detailed description of the desired image content, in English',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False,
        )


def init_agent_service():
    llm_cfg = {"generate_cfg": {"max_input_tokens": 31000}}
    system = "你是我的私人助理。你的回答要风趣幽默，口语化，有批判性，尖锐，不要总是用列表式的回答来糊弄我，善于用一些例子来表达你的观点。"

    tools = [
        'my_image_gen',
        'code_interpreter',
        'web_search'
    ]  # code_interpreter is a built-in tool in Qwen-Agent
    bot = Assistant(
        llm=llm_cfg,
        name='助理',
        description='我的私人助理',
        system_message=system,
        function_list=tools,
        files=[],
    )

    return bot


def app_test(query: str = "draw a software architecture diagram, user's text input to LLM, and LLM calls function calling, with graphrag"):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = [
        {'role': 'user', 'content': "画一只猫"},
        {'role': 'assistant', 'content': "好的，这就为你画"},
        {'role': 'user', 'content': "旁边有条狗"},
        {'role': 'assistant', 'content': "好的，这就为你画"},
        {'role': 'user', 'content': "查询东方国信的相关资讯"}
    ]
    for response in bot.run(messages=messages):
        print('bot response:', response)


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [
            '画一只猫的图片',
            '画一只可爱的小腊肠狗',
            '画一幅风景画，有湖有山有树',
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == '__main__':
    # app_test()
    # app_tui()
    app_gui()
