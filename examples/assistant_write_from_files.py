import json
import os

import pandas as pd
import pymed
from modelscope_studio.components.Chatbot import ModelScopeChatbot
from pymed.article import PubMedArticle

from qwen_agent import Agent
from qwen_agent.agents import Assistant, ArticleAgent
from qwen_agent.agents.medical.pubmed_agent import PubMedAgent
from qwen_agent.agents.write_from_files import WriteFromFiles
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import USER, Message
from qwen_agent.tools.simple_doc_parser import parse_pdf
from qwen_agent.utils.global_knowledge_base import GlobalKnowledgeBase
from qwen_agent.utils.tokenization_qwen import count_tokens


def app_test():
    files = [entry.path for entry in os.scandir(r"f:\resources\stroke")]
    bot = WriteFromFiles(
        llm={'model': 'qwen2-72b-instruct'},
        files=files,
        rag_cfg={'max_ref_tokens': 20000})

    messages = [Message(USER, 'a summary of the current status and development trends in pediatric stroke research')]
    # {'role': 'user', 'content': [{'text': '儿童卒中研究现状及发展综述'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def get_paper_meta(record: dict, agent: Agent) -> dict:
    print(f"processing {record['pmid']}")

    pdf_file = fr"f:\resources\stroke\{record['pmid']}.pdf"
    if not os.path.exists(pdf_file):
        return None

    json_file = fr"f:\resources\stroke_text\{record['pmid']}.meta4.json"
    if os.path.exists(json_file):
       return json.load(open(json_file, 'r', encoding='utf8'))

    pubmed = pymed.PubMed()
    results = list(pubmed.query("PMID:" + str(record['pmid']), 1))
    article: PubMedArticle = results[0]
    article_pubmed_id = article.pubmed_id.split("\n")[0]
    assert int(article_pubmed_id) == int(record['pmid']), f"PMID mismatch {article.pubmed_id}:{record['pmid']}"

    if len(article.authors) > 0:
        first_author = article.authors[0]['firstname'] + " " + article.authors[0]['lastname']
    else:
        first_author = ''

    lines = []
    contents = parse_pdf(pdf_file)
    for content in contents:
        lines.extend([c['text'].strip(" \n") for c in content['content'] if 'text' in c])

    raw_text = "\n".join(lines)
    lines = raw_text.split("\n")
    raw_text = "\n".join([line for line in lines if len(line) >= 5])

    prompt = """You are a writing assistant, please summarize the reference paper.
    
# Reference paper:
{ref_doc}


# Rules
The summary should use following format (don't use markdown format, just plain text):
demographics: - study demographics, for example, sample size, population characteristics, etc
start and end time:  - time frame of study
background: - study background
results: - study results
conclusions: - study conclusions
the importance of study: - why this study is important?


Summary:""".format(ref_doc=raw_text)

    *_, last = agent.run([Message(USER, prompt)], temperature=0)
    summary = last[0].content

    meta = {
        'title': article.title,
        'journal': article.journal,
        'first_author': first_author,
        'date_of_publication': str(article.publication_date),
        'summary': summary
    }
    json.dump(meta, open(json_file, 'w', encoding='utf8'), ensure_ascii=False)

    return meta


def app_gui():
    # df = pd.read_excel('f:\\resources\\stroke.xlsx')
    # df.sort_values(by='pubmed_pubdate', ascending=False, inplace=True)
    # df = df[df['impact_factor'] >= 10.0]
    #
    # summarize_agent = Assistant()
    # global_kb = GlobalKnowledgeBase()
    #
    # files = []
    # for index, record in df.iterrows():
    #     record = record.to_dict()
    #     meta = get_paper_meta(record, summarize_agent)
    #     if meta is None:
    #         continue
    #
    #     content = "\n".join([f"{key}: {value}" for key, value in meta.items()])
    #     files.append(global_kb.add_knowledge(content))

    # bot = ArticleAgent(llm={'model': 'qwen2-72b-instruct'}, record_formats=['html'])
    files = [entry.path for entry in os.scandir(r"D:\Temp\羌族")]
    bot = WriteFromFiles(
        llm={
            'model': 'qwen2-72b-instruct',
            'generate_cfg': {
                'presence_penalty': 1.5,
                'frequency_penalty': 1.5,
                "max_input_tokens": 160800
            }
        },
        files=files,
        rag_cfg={'max_ref_tokens': 20000})

    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '我的论文题目是"羌族释比艺术研究",你的身份是艺术学理论专家，请参考文献，梳理关于列维斯特劳斯在不同时期学术背景下，对他核心概念的理解，结合羌族原始宗教艺术——释比艺术，从人类学的视角分析羌族释比艺术的形成、文化内涵、艺术特征。'
            },
            {
                'text': 'a summary of the current status and development trends in pediatric stroke research'
            }
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # app_test()
    app_gui()
