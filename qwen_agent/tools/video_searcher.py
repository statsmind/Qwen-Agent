import json
import os
from typing import Union, List

import requests

from qwen_agent.llm.schema import ContentItem, Message
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool
from qwen_agent.utils.utils import has_chinese_chars


@register_tool("video_searcher")
class VideoSearcher(BaseTool):
    description = '视频搜索'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': '搜索关键字，使用关键字原来的语言，不需要转化',
            'required': True
        }
    ]

    def call(self, params: Union[str, dict], **kwargs) -> List[dict]:
        params = self._verify_json_format_args(params)
        query = params.get("query", "").strip()
        data = requests.get(
            f"https://api.bilibili.com/x/web-interface/search/all/v2?keyword={query}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT)",
                "Cookie": "SESSDATA=xxx"
            }
        ).json()

        video_list = []
        for item in data['data']['result']:
            if item['result_type'] == 'video':
                video_list = item['data']

        data = [
            {
                "video_link": item['arcurl'],
                'typename': item['typename'],
                'title': item['title'],
                'description': item['description']
            } for item in video_list
        ]

        return data
