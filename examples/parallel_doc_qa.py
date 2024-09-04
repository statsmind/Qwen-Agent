from qwen_agent.agents import Assistant
from qwen_agent.agents.doc_qa import ParallelDocQA, BasicDocQA
from qwen_agent.gui import WebUI


def test():
    bot = Assistant(llm={'model': 'qwen2-72b-instruct', 'generate_cfg': {'max_retries': 10, 'max_input_tokens': 31000}})
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'text': '这篇分析报告的观点是什么？'
                },
                {
                    'file': 'https://finance.sina.com.cn/stock/yyyj/2024-08-30/doc-incmkqct9220570.shtml'
                }
                # {
                #     'file': 'https://finance.sina.com.cn/roll/2024-08-31/doc-incmpiyf9400556.shtml'
                # },
                # {
                #     'file': 'https://finance.sina.com.cn/stock/yyyj/2024-08-31/doc-incmnxku2769167.shtml'
                # },
                # {
                #     'file': 'https://finance.sina.com.cn/jjxw/2024-08-31/doc-incmntap9659200.shtml'
                # },
                # {
                #     'file': 'https://finance.sina.com.cn/stock/relnews/cn/2024-08-30/doc-incmmvxh3144422.shtml'
                # },
                # {
                #     'file': 'https://finance.sina.com.cn/stock/hkstock/ggscyd/2024-08-30/doc-incmmvww3255060.shtml'
                # },
                # {
                #     'file': 'https://finance.sina.com.cn/stock/relnews/cn/2024-08-30/doc-incmmvww3249464.shtml'
                # }
            ]
        },
    ]
    for rsp in bot.run(messages):
        print('bot response:', rsp)


def app_gui():
    # Define the agent
    bot = BasicDocQA(
        llm={
            'model': 'qwen2-72b-instruct',
            'generate_cfg': {
                'max_retries': 10
            }
        },
        description='并行QA后用RAG召回内容并回答。支持文件类型：PDF/Word/PPT/TXT/HTML。使用与材料相同的语言提问会更好。',
    )

    chatbot_config = {'prompt.suggestions': [{'text': '介绍实验方法'}]}

    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    app_gui()
