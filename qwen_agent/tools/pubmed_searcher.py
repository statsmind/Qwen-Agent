import json
import os
from typing import Union, List

import requests

from qwen_agent.llm.schema import ContentItem, Message
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool
from qwen_agent.utils.utils import has_chinese_chars
from pymed import PubMed


@register_tool("pubmed_searcher")
class PubMedSearcher(BaseTool):
    description = 'search pubmed medical papers'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'keyword, all must be in English',
            'required': True
        },
        {
            'name': 'max_results',
            'type': 'int',
            'description': 'max results to be returned',
            'default': '100',
            'required': False
        },
    ]
    DEFAULT_MAX_RESULTS: int = 100

    def call(self, params: Union[str, dict], **kwargs) -> List[dict]:
        params = self._verify_json_format_args(params)

        query = params.get("query", "").strip()
        max_results = int(params.get("max_results", self.DEFAULT_MAX_RESULTS))

        papers = []
        for paper in PubMed().query(query, max_results=max_results):
            paper = paper.toDict()
            paper['publication_date'] = str(paper['publication_date'])
            paper['year'] = paper['publication_date'][:4]

            for field in ['pubmed_id', 'doi']:
                if field in paper and paper[field]:
                    parts = paper[field].split('\n')
                    if len(parts) > 0:
                        paper[field] = parts[0].strip()
                    else:
                        paper[field] = None
                else:
                    paper[field] = None

            if 'authors' in paper:
                paper['authors'] = [f"{author['firstname']} {author['lastname']} " for author in paper['authors']]
            else:
                paper['authors'] = []

            if len(paper['authors']) > 0:
                paper['first_author'] = paper['authors'][0]
            else:
                paper['first_author'] = " "

            for field in ['xml', 'copyrights']:
                if field in paper:
                    del paper[field]

            papers.append(paper)

        return papers
