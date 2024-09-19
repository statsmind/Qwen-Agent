import json
import os
from typing import Union, List

import requests

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool
from qwen_agent.utils.utils import has_chinese_chars


@register_tool("web_search")
class WebSearch(BaseTool):
    description = '根据关键字搜索相关网页'
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

        serp_url = "https://google.serper.dev/search"
        serp_api_key = os.environ.get("SERP_API_KEY", "")
        if not serp_api_key:
            raise ValueError("SERP_API_KEY env variable not set")

        query = params.get("query", "").strip()
        if not query:
            return []

        payload = {"q": query, "num": 30, "gl": "cn", "hl": "zh-cn"}
        if not has_chinese_chars(query):
            payload = {"q": query, "num": 30, "gl": "en", "hl": "en-us"}

        date_range = params.get("date_range", "")
        if date_range in ["y", "m", "d"]:
            payload["tbs"] = f"qdr:{date_range}"

        payload = json.dumps(payload, ensure_ascii=False)
        headers = {'X-API-KEY': serp_api_key, 'Content-Type': 'application/json'}
        try:
            response: dict = requests.request("POST", serp_url, headers=headers, data=payload, timeout=10).json()

            if 'organic' in response and response['organic']:
                files = [item['link'] for item in response['organic']]
            else:
                files = []

            return [ContentItem(file=file) for file in files]
        except:
            return []
