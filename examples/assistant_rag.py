from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools import WebSearcher, PubMedSearcher, VideoSearcher


def app_test():
    bot = Assistant(llm={'model': 'qwen2-72b-instruct'})
    messages = [{'role': 'user', 'content': [{'text': '介绍图一'}, {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def app_gui():
    # 第一步：构建知识库
    searcher = WebSearcher()

    links = []

    links.extend(searcher.call(params={'query': '儿童罕见病的现状、挑战及对策'}, num_results=90, cache=True))
    links.extend(searcher.call(params={'query': '儿童罕见病 特点 家庭的影响和负担'}, num_results=90, cache=True))
    links = list(set(links))

    # Define the agent
    bot = Assistant(llm={'model': 'qwen2.5-72b-instruct'},
                    name='Assistant',
                    description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。',
                    files=links,
                    record_formats=['jsonl', 'html']
                    )
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '儿童罕见病的现状、挑战及对策'
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
