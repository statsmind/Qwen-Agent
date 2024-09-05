from dotenv import load_dotenv

from qwen_agent.agents import Assistant, ArticleAgent
from qwen_agent.gui import WebUI
from qwen_agent.utils.utils import list_files


def app_test():
    files = list_files('F:\\resources\\sci-hub')[:5]
    bot = ArticleAgent(llm={}, files=files)
    messages = [{'role': 'user', 'content': 'BCI history'}]
    for rsp in bot.run(messages, full_article=True):
        print(rsp)


def app_gui():
    # Define the agent
    agent = ArticleAgent(
        llm={'model': 'qwen2-72b-instruct', 'generate_cfg': {
            'max_input_tokens': 31000
        }},
        name='写作助手',
        description='论文，发言稿，文书写作',
        files=[
            r'C:\Users\james\Desktop\关于开展庆祝中华人民共和国成立75周年主题征文活动的通知.doc',
            'https://www.ngd.org.cn/qmtc/rmzxb/16f798ab4d0f4643b3d2ea038633e241.htm',
            'https://www.ngd.org.cn/xwzx/ywdt/17ae49f04ac348839080738b2c1a4a07.htm',
            'https://www.ngd.org.cn/xwzx/ywdt/e7ef191a6759404faa8047d8476fd592.htm',
            'https://www.ngd.org.cn/xwzx/ywdt/4d80847832524b54be92f88b87d46883.htm',
            'https://www.ngd.org.cn/xwzx/ywdt/b558f44c5f95415ab122c95ca2355546.htm',
            'https://www.ngd.org.cn/xwzx/ywdt/b4e0a7a8938943709baf43018e781d49.htm',
            'https://www.ngd.org.cn/zsjs1/sxll/30793f6b191f484ab562a679ab080b52.htm',
            'https://www.ngd.org.cn/zsjs1/sxll/6f23da723bc244ba8be5cc9ee1265845.htm',
            'https://www.ngd.org.cn/dyfc/5b0cd62d22ed4aaa84dda6556dd6c2eb.htm',
            'https://www.ngd.org.cn/dyfc/3054df4a0c1b46b3a319cd2ea5d59078.htm',
            'https://www.ngd.org.cn/dyfc/c073a13dbc2d4c49b51410e6c762a4a1.htm',
            'https://www.ngd.org.cn/dyfc/bac176916593420fb6752b4ed423eecf.htm',
            'https://www.ngd.org.cn/dyfc/248960e2b2c848038b3c4fa4a97f57e1.htm'
        ]
    )
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
