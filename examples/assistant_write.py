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



def app_gui():

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
                'text': '儿童卒中的研究进展'
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
