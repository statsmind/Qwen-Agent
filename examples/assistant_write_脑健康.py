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
from qwen_agent.tools import WebSearcher
from qwen_agent.tools.simple_doc_parser import parse_pdf
from qwen_agent.utils.global_knowledge_base import GlobalKnowledgeBase
from qwen_agent.utils.tokenization_qwen import count_tokens


def app_gui():
    # 第一步：构建知识库
    # searcher = WebSearcher()
    #
    # links = searcher.call(params={'query': '脑健康 科技创新 产业发展'}, num_results=90, cache=True)
    # links = list(set(links))

    bot = Assistant(
        llm={
            'model': 'qwen2.5-72b-instruct',
            'generate_cfg': {
                'temperature': 0.5,
                'max_input_tokens': 120500
            }
        },
        record_formats=['html'])

    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '提案：脑健康科技创新与产业发展'
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


def app_test():
    # kbs = json.load(open('workspace/kb.json', 'r', encoding='utf-8'))
    #
    # system_prompt = ""
    # for index, kb in enumerate(kbs):
    #     system_prompt += f"## doc_{index}\n{kb['text'][0]}\n\n"
    # print(system_prompt)

    pass

if __name__ == '__main__':
    # app_test()
    app_gui()
