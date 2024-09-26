"""A multi-agent cooperation example implemented by router and assistant"""

import os
from typing import Optional

import pandas as pd

from qwen_agent.agents import Assistant, ReActChat, Router
from qwen_agent.gui import WebUI
import json
import os

import pandas as pd
import pymed
from modelscope_studio.components.Chatbot import ModelScopeChatbot
from pymed.article import PubMedArticle

from qwen_agent import Agent
from qwen_agent.agents import Assistant, ArticleAgent
from qwen_agent.agents.assistant import KNOWLEDGE_SNIPPET, KNOWLEDGE_TEMPLATE
from qwen_agent.agents.medical.pubmed_agent import PubMedAgent
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import USER, Message
from qwen_agent.tools.simple_doc_parser import parse_pdf
from qwen_agent.utils.global_knowledge_base import GlobalKnowledgeBase
from qwen_agent.utils.tokenization_qwen import count_tokens


ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


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


def get_paper_etiological(record: dict, agent: Agent) -> dict:
    print(f"processing {record['pmid']}")

    pdf_file = fr"f:\resources\stroke\{record['pmid']}.pdf"
    if not os.path.exists(pdf_file):
        return None

    json_file = fr"f:\resources\stroke_text\{record['pmid']}.meta5.json"
    if os.path.exists(json_file):
        return json.load(open(json_file, 'r', encoding='utf8'))

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
The summary should only contains `etiological composition` and `gene`, use following format (don't use markdown format, just plain text):
etiological composition: -- list all etiologies mentioned in the paper, including prevalence of causes, mortality and disability
gene: -- genetic testing/gene sequencing/gene therapy

Summary:""".format(ref_doc=raw_text)

    *_, last = agent.run([Message(USER, prompt)], temperature=0)
    summary = last[0].content

    meta = {
        'summary': summary
    }
    json.dump(meta, open(json_file, 'w', encoding='utf8'), ensure_ascii=False)

    return meta



def init_agent_service():
    # Define a vl agent
    bot_vl = Assistant(
        llm={'model': 'qwen-vl-max'},
        name='多模态助手',
        description='可以理解图像内容。')

    # Define a tool agent
    bot_tool = Assistant(
        llm={'model': 'qwen2.5-72b-instruct'},
        name='网页搜索助手',
        description='使用网页搜索来解决问题',
        function_list=['web_searcher'],
    )

    pubmed_tool = Assistant(
        llm={'model': 'qwen2.5-72b-instruct'},
        name='PUBMED助手',
        description='使用pubmed搜索来解决问题',
        function_list=['pubmed_searcher'],
    )

    df = pd.read_excel('f:\\resources\\stroke.xlsx')
    df.sort_values(by='pubmed_pubdate', ascending=False, inplace=True)
    df = df[df['impact_factor'] >= 10.0]

    summarize_agent = Assistant()

    snippets = []
    for index, record in df.iterrows():
        record = record.to_dict()
        meta = get_paper_meta(record, summarize_agent)
        if meta is None:
            continue

        meta2 = get_paper_etiological(record, summarize_agent)
        meta['summary'] = meta['summary'] + "\n" + meta2['summary']

        content = "\n".join([f"{key}: {value}" for key, value in meta.items()])
        snippet = KNOWLEDGE_SNIPPET['en'].format(source=f'paper{record["pmid"]}', content=content)
        snippets.append(snippet)

    knowledge = "\n\n".join(snippets)

    child_stroke_bot = ArticleAgent(
        llm={
            'model': 'qwen2.5-72b-instruct',
            'generate_cfg': {
                'temperature': 0.5,
                'max_input_tokens': 120500
            }
        },
        name='儿童卒中论文助手',
        description='帮你写儿童卒中论文相关论文',
        record_formats=['html'],
        knowledge=knowledge)

    child_stroke_ask = Assistant(
        llm={
            'model': 'qwen2.5-72b-instruct',
            'generate_cfg': {
                'temperature': 0.5,
                'max_input_tokens': 120500
            }
        },
        name='儿童卒中问答',
        description='儿童卒中问答',
        record_formats=['html'],
        system_message=f"You are helpful assistant. You can access following knowledge base to answer my question.\n# Knowledge Base\n{knowledge}")

    # Define a router (simultaneously serving as a text agent)
    bot = Router(
        llm={'model': 'qwen2.5-72b-instruct'},
        agents=[child_stroke_bot, child_stroke_ask, bot_vl, bot_tool, pubmed_tool],
    )
    return bot


def app_test(
        query: str = 'hello',
        image: str = 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg',
        file: Optional[str] = os.path.join(ROOT_RESOURCE, 'poem.pdf'),
):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []

    if not image and not file:
        messages.append({'role': 'user', 'content': query})
    else:
        messages.append({'role': 'user', 'content': [{'text': query}]})
        if image:
            messages[-1]['content'].append({'image': image})
        if file:
            messages[-1]['content'].append({'file': file})

    for response in bot.run(messages):
        print('bot response:', response)


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        query = input('user question: ')
        # Image example: https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg
        image = input('image url (press enter if no image): ')
        # File example: resource/poem.pdf
        file = input('file url (press enter if no file): ').strip()
        if not query:
            print('user question cannot be empty！')
            continue
        if not image and not file:
            messages.append({'role': 'user', 'content': query})
        else:
            messages.append({'role': 'user', 'content': [{'text': query}]})
            if image:
                messages[-1]['content'].append({'image': image})
            if file:
                messages[-1]['content'].append({'file': file})

        response = []
        for response in bot.run(messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    bot = init_agent_service()
    chatbot_config = {
        'verbose': True,
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui()
