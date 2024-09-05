from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI


def app_test():
    bot = Assistant(llm={'model': 'qwen2-72b-instruct'}, function_list=['web_search'])
    messages = [{'role': 'user', 'content': [{'text': 'angular路由是怎么工作的？举个实际的例子'}, {'file': 'D:\\workspace\\gitee\\NiceFish'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def app_gui():
    # Define the agent
    bot = Assistant(llm={'model': 'qwen2-72b-instruct'}, function_list=['web_search'],
                    name='Assistant',
                    description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。',
                    files=['D:\\workspace\\gitee\\NiceFish']
                    )
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': 'angular路由是怎么工作的？举个实际的例子'
            },
            {
                'text': '第二章第一句话是什么？'
            },
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # app_test()
    app_gui()
