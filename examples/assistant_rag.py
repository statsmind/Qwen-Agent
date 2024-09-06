from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI


def app_test():
    bot = Assistant(
        function_list=['web_search', 'code_interpreter'],
        files=[
            r'D:\workspace\mine\medical2.0\openhis-api\src',
            r'D:\workspace\mine\medical2.0\openhis-ui\src'
        ]
    )
    messages = [{'role': 'user', 'content': [{'text': '加入功能：药品可以转移到其他药库，给出前后端代码实现，以及代码的位置'}]}]
    for rsp in bot.run(messages):
        print(rsp)

    print("=========================================")
    for item in rsp:
        print(item['content'])


def app_gui():
    # Define the agent
    bot = Assistant(llm={'model': 'qwen2-72b-instruct'},
                    function_list=['web_search', 'code_interpreter'],
                    name='Assistant',
                    description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。',
                    files=[
                        r'D:\workspace\mine\medical2.0\openhis-api\src',
                        r'D:\workspace\mine\medical2.0\openhis-ui\src'
                    ]
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
    app_test()
    # app_gui()
