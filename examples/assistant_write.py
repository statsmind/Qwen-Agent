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
from qwen_agent.llm.schema import USER, Message, ASSISTANT
from qwen_agent.recorder.message_recorder import MessageRecorder
from qwen_agent.tools.simple_doc_parser import parse_pdf
from qwen_agent.utils.global_knowledge_base import GlobalKnowledgeBase
from qwen_agent.utils.tokenization_qwen import count_tokens


def app_test():
    global_kb = GlobalKnowledgeBase()
    files = []
    for entry in os.scandir(r"f:\resources\stroke_text"):
        if entry.name.endswith("meta2.json"):
            paper = json.load(open(entry.path, 'r', encoding='utf8'))
            if 'abstract' in paper:
                del paper['abstract']

            content = "\n".join([f"{key}: {value}" for key, value in paper.items()])
            files.append(global_kb.add_knowledge(content))

    bot = ArticleAgent(llm={'model': 'qwen2-72b-instruct'}, files=files, record_formats=['html'])
    messages = [{'role': 'user', 'content': [{'text': '儿童卒中研究现状及发展综述'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def get_paper_meta(record: dict, agent: Agent) -> dict:
    print(f"processing {record['pmid']}")

    pdf_file = fr"f:\resources\stroke\{record['pmid']}.pdf"
    if not os.path.exists(pdf_file):
        return None

    json_file = fr"f:\resources\stroke_text\{record['pmid']}.meta6.json"
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
The summary must strictly adhere to the content of the provided Knowledge Base, even if it deviates from the facts. Use following format (don't use markdown format, just plain text):
demographics: - study demographics, for example, sample size, population characteristics, etc
start and end time:  - time frame of study
background: - study background
results: - study results
conclusions: - study conclusions
etiologies: - list all etiologies mentioned in the paper, including prevalence of causes, mortality and disability
gene: - genetic testing/gene sequencing/gene therapy
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
    df = pd.read_excel('f:\\resources\\stroke.xlsx')
    df.sort_values(by='pubmed_pubdate', ascending=False, inplace=True)
    df = df[df['impact_factor'] >= 10.0]

    summarize_agent = Assistant()
    message_recorder = MessageRecorder(output_formats=['html'])
    message_recorder.start(Message(USER, "References"))
    record_messages = []

    snippets = []
    for index, record in df.iterrows():
        record = record.to_dict()
        meta = get_paper_meta(record, summarize_agent)
        if meta is None:
            continue

        content = "\n".join([f"{key}: {value}" for key, value in meta.items()])
        snippet = KNOWLEDGE_SNIPPET['en'].format(source=f'paper{record["pmid"]}', content=content)
        snippets.append(snippet)
        record_messages.append(Message(ASSISTANT, f"#{record['pmid']}\n<pre style='overflow-x: auto; white-space: pre-wrap; white-space: -moz-pre-wrap; white-space: -pre-wrap; white-space: -o-pre-wrap; word-wrap: break-word;'>{content}</pre>"))
    message_recorder.step(record_messages, True)

    knowledge = "\n\n".join(snippets)

    bot = ArticleAgent(
        llm={
            'model': 'qwen2.5-72b-instruct',
            'generate_cfg': {
                'temperature': 0.5,
                'max_input_tokens': 120500
            }
        },
        record_formats=['html'],
        knowledge=knowledge)
    # bot = Assistant(
    #     llm={
    #         'model': 'qwen2.5-72b-instruct',
    #         'generate_cfg': {
    #             'temperature': 0.5,
    #             'max_input_tokens': 120500
    #         }
    #     },
    #     record_formats=['html'],
    #     system_message=f"You are helpful assistant. You can access following knowledge base to answer my question.\n# Knowledge Base\n{knowledge}")
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '儿童卒中研究现状及发展综述'
            },
            {
                'text': 'A review of the current status and development of pediatric stroke research'
            },
            {
                'text': 'Etiology of pediatric stroke and the corresponding morbidity, mortality and disability rates'
            },
            {
                'text': 'Gene therapy for pediatric stroke'
            }
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # app_test()
    app_gui()
