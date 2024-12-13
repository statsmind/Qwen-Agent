import json
import os
from typing import Union, List

import requests

from qwen_agent.llm.schema import ContentItem, Message
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool
from qwen_agent.utils.cache import Cache
from qwen_agent.utils.utils import has_chinese_chars


@register_tool("web_searcher")
class WebSearcher(BaseTool):
    description = 'web search'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': 'search keywords',
            'required': True
        },
        {
            'name': 'lang',
            'type': 'string',
            'description': 'the language',
            "enum": ["en", "zh", "auto"],
            "default": "auto",
            'required': False
        },
        {
            'name': 'date_range',
            'type': 'string',
            'description': '搜索时间范围, y: 最近一年; m: 最近一个月; d: 最近一天; 空表示无时间限制',
            "enum": ["y", "m", "d", ""],
            'default': '',
            'required': False
        },
    ]
    serp_url = "https://google.serper.dev/search"
    empty_response = {
        "searchParameters": {},
        "knowledgeGraph": None,
        "organic": [],
        "peopleAlsoAsk": [],
        "relatedSearches": [],
        "credits": 2
    }

    def call(self, params: Union[str, dict], num_results: int = 30, cache: bool = True, **kwargs) -> List:
        params = self._verify_json_format_args(params)

        serp_api_key = os.environ.get("SERP_API_KEY", "")
        if not serp_api_key:
            raise ValueError("SERP_API_KEY env variable not set")

        query = params.get("query", "").strip()
        lang = params.get("lang", "auto")

        payload = {"q": query, "num": num_results, "gl": "cn", "hl": "zh-cn"}
        if lang == 'en' or (lang == 'auto' and not has_chinese_chars(query)):
            payload['gl'] = 'en'
            payload['hl'] = 'en-us'

        if query:
            date_range = params.get("date_range", "")
            if date_range in ["y", "m", "d"]:
                payload["tbs"] = f"qdr:{date_range}"

            payload = json.dumps(payload, ensure_ascii=False)

            cache_api = Cache("web_searcher") if cache else None
            if cache_api is not None:
                content = cache_api.get(payload)
                if content is not None:
                    response = json.loads(content)
                    return [item['link'] for item in response['organic']]

            headers = {'X-API-KEY': serp_api_key, 'Content-Type': 'application/json'}
            try:
                response: dict = requests.request(
                    "POST", self.serp_url, headers=headers, data=payload, timeout=10
                ).json()

                if 'organic' in response:
                    if cache_api is not None:
                        cache_api.put(payload, json.dumps(response, ensure_ascii=False))

                    return [item['link'] for item in response['organic']]
            except:
                pass

        #return {**self.empty_response, "searchParameters": payload}
        return []
