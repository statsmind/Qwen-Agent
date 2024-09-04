import json
import os
from typing import Union, List

import requests

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool


@register_tool("web_search")
class WebSearch(BaseTool):
    description = '根据关键字搜索相关信息'
    parameters = [{'name': 'query', 'type': 'string', 'description': '搜索关键字', 'required': True}]

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, List[ContentItem]]:
        params = self._verify_json_format_args(params)

        serp_url = "https://google.serper.dev/search"
        serp_api_key = os.environ["SERP_API_KEY"]

        payload = json.dumps({
            "q": params.get("query", ""),
            "gl": "cn",
            "hl": "zh-cn"
        })
        headers = {
            'X-API-KEY': serp_api_key,
            'Content-Type': 'application/json'
        }

        result = []
        response: dict = requests.request("POST", serp_url, headers=headers, data=payload).json()

        if 'organic' in response and response['organic']:
            for item in response['organic']:
                item = {
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'date': item.get('date', ''),
                    'link': item.get('link', '')
                }
                result.append(item)

        files = [item['link'] for item in result]
        return [ContentItem(file=file) for file in files]
