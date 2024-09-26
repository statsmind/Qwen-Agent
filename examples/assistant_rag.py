from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools import WebSearcher, PubMedSearcher, VideoSearcher


def app_test():
    bot = Assistant(llm={'model': 'qwen2-72b-instruct'})
    messages = [{'role': 'user', 'content': [{'text': '介绍图一'}, {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def app_gui():
    # Define the agent
    bot = Assistant(llm={'model': 'qwen2.5-72b-instruct'},
                    name='Assistant',
                    description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。',
                    function_list=[WebSearcher(), PubMedSearcher(), VideoSearcher()],
                    files=[
                        r'C:\Users\james\Documents\you-et-al-2024-twenty-four-hour-post-thrombolysis-nihss-score-as-the-strongest-prognostic-predictor-after-acute.pdf'
                    ],
                    record_formats=['jsonl', 'html']
                    )
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '这篇文章研究设计和统计学方法上有什么特点？'
            },
            {
                'text': '这篇文章有什么特点？'
            },
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    app_gui()
