import json
from typing import Iterator, List, Optional, Union, Dict

from qwen_agent import Agent
from qwen_agent.agents.assistant import Assistant
from qwen_agent.agents.keygen_strategies import SplitQueryThenGenKeyword
from qwen_agent.agents.write_from_scratch import WriteFromScratch
from qwen_agent.agents.writing import ContinueWriting
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, CONTENT, Message, DEFAULT_SYSTEM_MESSAGE
from qwen_agent.tools import BaseTool, PubMedSearcher, WebSearcher
from qwen_agent.tools.doc_parser import Record, Chunk
from qwen_agent.utils.global_knowledge_base import GlobalKnowledgeBase
from qwen_agent.utils.tokenization_qwen import count_tokens


class PubMedAgent(Assistant):
    def _run(self,
             messages: List[Message],
             lang: str = 'en',
             **kwargs) -> Iterator[List[Message]]:

        keyword_gen = SplitQueryThenGenKeyword(llm=self.llm)
        *_, last = keyword_gen.run(messages, lang=lang, **kwargs)
        keywords = json.loads(last[0].content)

        pubmed_searcher = PubMedSearcher()
        local_kb = GlobalKnowledgeBase()
        web_searcher = WebSearcher()
        assistant = Assistant(llm=self.llm)

        keywords_en = []
        for keyword_zh in keywords['keywords_zh']:
            search_result = web_searcher.call({'query': f'{keyword_zh} pubmed', 'lang': 'en'})
            ref = [{'url': item['link'], 'text': item.get('snippet', '')} for item in search_result['organic'] if item.get('snippet', '')]

            *_, last = assistant.run([Message('user', f'{keyword_zh}的英文名称是什么？只回答英文名称，不要添加额外信息')], knowledge=ref)
            keywords_en.append(last[0].content)

        papers = []
        for keyword_en in keywords_en:
            papers.extend(pubmed_searcher.call({"query": keyword_en, "max_results": 100}))

        kb_files = []
        for paper in papers:
            content = json.dumps({
                'title': paper.get('title', ''),
                'publication_date': paper.get('publication_date', ''),
                'journal': paper.get('journal', ''),
                'abstract': paper.get('abstract', ''),
                'methods': paper.get('methods', ''),
                'conclusions': paper.get('conclusions', ''),
                'results': paper.get('results', '')
            }, ensure_ascii=False)
            kb_file_id = local_kb.add_knowledge([{
                'page_num': 1,
                'title': f"{paper['first_author']}, {paper['year']}",
                'content': [
                    {'text': content, 'token': count_tokens(content)}
                ]
            }])
            kb_files.append(kb_file_id)

        assistant = Assistant(files=kb_files, rag_cfg={
            "rag_searchers": ['keyword_search', 'vector_search'],
            "max_ref_token": 40000,
        })

        response = []
        for rsp in assistant.run(messages, lang=lang, **kwargs):
            if rsp:
                yield response + rsp
