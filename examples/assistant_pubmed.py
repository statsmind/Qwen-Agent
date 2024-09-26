from modelscope_studio.components.Chatbot import ModelScopeChatbot

from qwen_agent.agents import Assistant
from qwen_agent.agents.medical.pubmed_agent import PubMedAgent
from qwen_agent.gui import WebUI


def app_test():
    bot = Assistant(llm={'model': 'qwen2.5-72b-instruct'}, files=['f:\\resources\\stroke\\30355212.pdf'])
    messages = [{'role': 'user', 'content': 'Moyamoya病占儿童卒中的比例'}]
    for rsp in bot.run(messages):
        print(rsp)



def app_gui():
    # Define the agent
    bot = PubMedAgent(llm={'model': 'qwen2.5-72b-instruct'},
                      name='PUBMED助手',
                      description='使用PubMED检索并回答',
                      system_message="You are helpful assistant. When you answer my question, please add reference anchor to your reply and append References list to the end.",
                      record_formats=['jsonl', 'html'])
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '安宫牛黄丸具有抗炎和保护血脑屏障的作用，有动物或临床研究的证据吗？'
            },
            {
                'text': '复方丹参片具有抗炎和保护血脑屏障的作用，有动物或临床研究的证据？回答最后附带参考文献列表'
            },
            {
                'text': '华佗再造丸具有抗炎和保护血脑屏障的作用，有动物或临床研究的证据？回答最后附带参考文献列表'
            },
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    app_test()
    # app_gui()
