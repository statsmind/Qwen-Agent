import json
import os
from typing import Union, List
from pymed import PubMed
import requests

from qwen_agent.llm.schema import ContentItem, Message, USER
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool
from qwen_agent.utils.utils import last_item


@register_tool("pubmed_search")
class PubmedSearch(BaseTool):
    description = '根据关键字搜索Pubmed医学论文'
    parameters = [
        {'name': 'query', 'type': 'string', 'description': '搜索关键字', 'required': True},
        {
            'name': 'date_range',
            'type': 'string',
            'description': '搜索时间范围, y: 最近一年; m: 最近一个月; d: 最近一天; 空表示无时间限制',
            "enum": ["y", "m", "d", ""],
            'default': '',
            'required': False
        },
    ]

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, List[ContentItem]]:
        params = self._verify_json_format_args(params)

        query = params.get("query", "").strip()
        if not query:
            return []

        llm = params["llm"]

        content_items = []
        pubmed = PubMed(tool="MyTool", email="jameshu@live.com")
        results = pubmed.query(query, max_results=100)

        for result in results:
            if not result.pubmed_id:
                continue

            pubmed_id = result.pubmed_id.split("\n")[0]
            en_content = f"""Title: {result.title}\nPublication Date: {str(result.publication_date)}\nAbstract: {result.abstract}\nMethods: {result.methods or ""}\nResults: {result.results or ""}\nConclusions: {result.conclusions or ""}"""

            cache_en_file = f"F:\\datasets\\pubmed\\{pubmed_id}.txt"
            if not os.path.exists(cache_en_file):
                with open(cache_en_file, 'w', encoding='utf8') as fp:
                    fp.write(en_content)

            if os.path.exists(cache_en_file) and os.path.getsize(cache_en_file) > 0:
                content_items.append(ContentItem(file=cache_en_file))

            cache_cn_file = f"F:\\datasets\\pubmed\\{pubmed_id}_cn.txt"
            if not os.path.exists(cache_cn_file) or os.path.getsize(cache_cn_file) == 0:
                response = last_item(llm.chat(messages=[
                    Message(USER, f"翻译下面的文本:\n----------------\n{en_content}")
                ]))
                with open(cache_cn_file, 'w', encoding='utf8') as fp:
                    fp.write(response[0].content)

            if os.path.exists(cache_cn_file) and os.path.getsize(cache_cn_file) > 0:
                content_items.append(ContentItem(file=cache_cn_file))

        return content_items


if __name__ == '__main__':
    search = PubmedSearch()
    search.call({"query": "stroke"})

