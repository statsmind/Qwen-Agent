import json
import os
from typing import Union, List

import requests

from qwen_agent.llm.schema import ContentItem, Message
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool
from qwen_agent.utils.utils import has_chinese_chars


@register_tool("web_searcher")
class WebSearcher(BaseTool):
    description = '根据关键字搜索相关网页'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': '搜索关键字',
            'required': True
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

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        serp_api_key = os.environ.get("SERP_API_KEY", "")
        if not serp_api_key:
            raise ValueError("SERP_API_KEY env variable not set")

        query = params.get("query", "").strip()
        payload = {"q": query, "num": 30, "gl": "cn", "hl": "zh-cn"}
        if not has_chinese_chars(query):
            payload['gl'] = 'en'
            payload['hl'] = 'en-us'

        if query:
            date_range = params.get("date_range", "")
            if date_range in ["y", "m", "d"]:
                payload["tbs"] = f"qdr:{date_range}"

            payload = json.dumps(payload, ensure_ascii=False)
            headers = {'X-API-KEY': serp_api_key, 'Content-Type': 'application/json'}
            try:
                response: dict = requests.request(
                    "POST", self.serp_url, headers=headers, data=payload, timeout=10
                ).json()

                if 'organic' in response:
                    return json.dumps(response, ensure_ascii=False)
            except:
                pass

        return json.dumps({**self.empty_response, "searchParameters": payload}, ensure_ascii=False)


if __name__ == "__main__":
    from qwen_agent.agents import Assistant

    assistant = Assistant(
        function_list=[WebSearcher()]
    )

    *_, response = assistant.run([Message('user', '安宫牛黄丸动物或临床研究的证据')])
    print(response)