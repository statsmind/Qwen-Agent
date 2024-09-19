from modelscope_studio.components.Chatbot import ModelScopeChatbot

from qwen_agent.agents import Assistant
from qwen_agent.agents.medical.pubmed_agent import PubMedAgent
from qwen_agent.gui import WebUI


def app_test():
    bot = PubMedAgent(llm={'model': 'qwen2-72b-instruct'})
    messages = [{'role': 'user', 'content': [{'text': '介绍图一'}, {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]
    for rsp in bot.run(messages):
        print(rsp)

def app_gui():
    # Define the agent
    bot = PubMedAgent(llm={'model': 'qwen2-72b-instruct'},
                      name='Assistant',
                      description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。')
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '安宫牛黄丸动物或临床研究的证据'
            },
            {
                'text': '复方丹参片动物或临床研究的证据？'
            },
            {
                'text': '华佗再造丸动物或临床研究的证据？'
            },
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # app_test()
    app_gui()
