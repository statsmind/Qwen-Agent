import os

from dotenv import load_dotenv

from qwen_agent.agents import Assistant, ArticleAgent
from qwen_agent.gui import WebUI
from qwen_agent.llm import get_chat_model
from qwen_agent.tools import WebSearch
from qwen_agent.tools.pubmed_search import PubmedSearch
from qwen_agent.utils.utils import list_files


def app_test():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "true"

    web_search = WebSearch()
    content_items = web_search.call({"query": "pediatric stroke cohort"})
    content_items += web_search.call({"query": "儿童卒中队列的建设"})
    content_items += web_search.call({"query": "youth stroke cohort"})
    content_items += web_search.call({"query": "青年卒中队列的建设"})

    pubmed_search = PubmedSearch()
    content_items += pubmed_search.call({"query": "pediatric stroke cohort"})
    content_items += pubmed_search.call({"query": "youth stroke cohort"})

    files = [item.file for item in content_items]
    bot = ArticleAgent(llm={"generate_cfg": {"max_input_tokens": 30500}}, files=files)
    messages = [{'role': 'user', 'content': '国内外儿童卒中/青年卒中队列的建设情况'}]
    for rsp in bot.run(messages, full_article=True):
        print(rsp)


def app_gui():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
    llm = get_chat_model({})

    # web_search = WebSearch()
    # content_items = web_search.call({"query": "+pediatric +stroke cohort"})
    # content_items += web_search.call({"query": "+儿童 +卒中 队列"})
    # content_items += web_search.call({"query": "+youth +stroke cohort"})
    # content_items += web_search.call({"query": "+青年 +卒中 队列"})

    # content_items = []
    # pubmed_search = PubmedSearch()
    # content_items += pubmed_search.call({"query": "(pediatric AND stroke) cohort", "llm": llm})
    # content_items += pubmed_search.call({"query": "(youth AND stroke) cohort", "llm": llm})
    #
    # files = [item.file for item in content_items]
    # files.extend([
    #         "https://rs.yiigle.com/CN2021/1477610.htm",
    #         "https://pdf.hanspub.org/ACM20240300000_96708515.pdf",
    #         "https://nursing.medsci.cn/article/show_article.do?id=7a592e81884",
    #         "https://rs.yiigle.com/CN2021/1477610.htm",
    #         "https://www.sohu.com/a/331992773_120051769",
    #         "https://www.medsci.cn/article/show_article.do?id=36415881636",
    #         "https://www.sohu.com/a/377692621_100294939",
    #         "http://neuro.dxy.cn/article/146867",
    #         "https://www.brainmed.com/info/detail?id=15896",
    #         "http://dohadchina.case.soogee.com/index.php?s=/Index/news_cont/id/519.html",
    #         "https://news-cdn.medlive.cn/all/info-progress/show-191157_100.html",
    #         "https://pdf.hanspub.org/ACM20220100000_98082333.pdf",
    #         "https://www.medsci.cn/article/show_article.do?id=e8fc29559487",
    #         "https://www.researchgate.net/publication/369639348_Research_Progress_in_Etiology_and_Risk_Factors_of_Ischemic_Stroke_in_Young_People/fulltext/6425cd6b315dfb4ccebc2aa3/Research-Progress-in-Etiology-and-Risk-Factors-of-Ischemic-Stroke-in-Young-People.pdf",
    #         "https://zhuanlan.zhihu.com/p/585612492",
    #         "https://new.qq.com/rain/a/20220808A08C6P00",
    #         "https://www.sohu.com/a/627679612_121123713",
    #         "https://www.uptodate.com/contents/zh-Hans/hemorrhagic-stroke-in-children",
    #         "https://www.sohu.com/a/742229338_121124519",
    #         "https://www.medsci.cn/article/show_article.do?id=a21f293486d9",
    #         "https://lifescience.sinh.ac.cn/webadmin/upload/20240108151849_3721_6217.pdf",
    # ])
    # files = [file for file in files if "1437056.htm" not in file]

    files = [
        "https://zhuanlan.zhihu.com/p/192446291",
        "https://www.sohu.com/a/419965708_120865534",
        "https://zhuanlan.zhihu.com/p/438043258",
        "https://www.biomedrxiv.org.cn/article/pdf/display/bmr.202007.00018",
        "https://www.biodiversity-science.net/article/2021/1005-0094/1005-0094-29-10-1425.shtml"
    ]

    # Define the agent
    agent = ArticleAgent(
        llm={
            'model': 'qwen2-72b-instruct',
            'generate_cfg': {
                'max_input_tokens': 31000
            }
        },
        rag_cfg={
            'max_ref_token': 20000,
            'parser_page_size': 250
        },
        name='写作助手',
        description='论文，发言稿，文书写作',
        files=files
    )
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '生物样本库如何建设？'
            }
        ]
    }
    WebUI(agent, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # app_test()
    app_gui()
