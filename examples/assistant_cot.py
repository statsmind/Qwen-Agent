from qwen_agent.agents import Assistant
from qwen_agent.agents.cot_agent import COTAgent
from qwen_agent.gui import WebUI
from qwen_agent.tools import WebSearcher, PubMedSearcher, VideoSearcher, CodeInterpreter


def app_test():
    bot = COTAgent(llm={'model': 'qwen2-72b-instruct'})
    messages = [{'role': 'user', 'content': 'strawberry 里面有几个r？'}]
    for rsp in bot.run(messages, lang='zh'):
        print(rsp)


def app_gui():
    # Define the agent
    bot = COTAgent(
        llm={'model': 'qwen2-72b-instruct'}
    )
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '0.9 或 0.11 哪个更大？'
            },
            {
                'text': 'strawberry 里面有几个r？'
            },
            {
                'text': '3307是质数吗？'
            },
            {
                'text': '3507是质数吗？'
            },
            {
                'text': '为什么我爸妈结婚的时候没有邀请我？'
            },
            {
                'text': '昨天的当天是明天的什么时间？'
            },
            {
                'text': '我的蓝牙耳机坏了，我该去看牙科还是耳鼻喉科？'
            },
            {
                'text': '每天吃一粒感冒药，还会感冒吗？'
            },
            {
                'text': '生鱼片是死鱼片吗？'
            },
            {
                'text': '等红灯是在等绿灯吗？'
            },
            {
                'text': '一个半小时是几个半小时？'
            }
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # app_test()
    app_gui()
