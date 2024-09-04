from dotenv import load_dotenv

from qwen_agent.agents import Assistant, ArticleAgent
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import list_files


def test():
    files = list_files('F:\\resources\\sci-hub')[:5]
    bot = ArticleAgent(llm={}, files=files)
    messages = [{'role': 'user', 'content': 'BCI history'}]
    for rsp in bot.run(messages, full_article=True):
        print(rsp)


def app_gui():
    # Define the agent
    agent = ArticleAgent(
        llm={'model': 'qwen2-72b-instruct'},
        name='写作助手',
        description='论文，发言稿，文书写作')
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '帮我写一篇爱国主义的发言稿'
            }
        ]
    }
    WebUI(agent, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    app_gui()
