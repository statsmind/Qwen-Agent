import json
from typing import Optional, Dict, List, Union
from urllib.parse import urljoin

import numpy as np
import requests
from jsonref import replace_refs

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools import BaseTool, HybridSearch
from qwen_agent.tools.base import ApiBaseTool, register_tool
from qwen_agent.tools.doc_parser import Record, Chunk


@register_tool("apibank")
class ApiBank(BaseTool):
    description = '获取外部工具列表'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '工具关键字',
        'default': '',
        'required': False
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> List[ApiBaseTool]:
        if isinstance(params, str):
            params = json.loads(params)
        query = params.get("query", "")

        api_dicts = self.parse_openapi_json("http://api.portal.clinify.cn/openapi.json")
        max_content_len = np.max([len(api_dict['name_for_human'] + ", " + api_dict['description']) for api_dict in api_dicts])
        max_query_repeat = (max_content_len + len(query) - 1) // len(query)
        max_query = "\n".join([query for _ in range(max_query_repeat)])

        api_funcs: Dict[str, ApiBaseTool] = self.create_functions(api_dicts)
        api_docs = [
            Record(
                url=api_dict['name'],
                title=api_dict['name_for_human'],
                raw=[
                    Chunk(
                        content=api_dict['name_for_human'] + ", " + api_dict['description'],
                        metadata={'name': api_dict['name'], 'source': api_dict['name'], 'chunk_id': 0},
                        token=10,
                    )
                ]
            )
            for index, api_dict in enumerate(api_dicts)]

        hybrid_search = HybridSearch(cfg={"rag_searchers": ['keyword_search', 'vector_search']})
        search_result = hybrid_search.search(json.dumps({"text": max_query}, ensure_ascii=False), api_docs, max_doc_num=5)
        return [api_funcs[result['url']] for result in search_result]

    @classmethod
    def create_functions(cls, api_dicts: List[Dict]) -> Dict[str, ApiBaseTool]:
        return {api_dict['name']: ApiBaseTool(cfg=api_dict) for api_dict in api_dicts}

    @classmethod
    def parse_openapi_json(cls, openapi_json_url: str) -> List[Dict]:
        api_dicts = []

        openapi_json = replace_refs(requests.get(openapi_json_url).json())
        for path, spec in openapi_json['paths'].items():
            name = path.replace("/", "_").strip(" _")
            schema = spec['post'] if 'post' in spec else spec['get']
            parameters = cls.extract_parameters(schema)

            api_dicts.append({
                'name': name,
                'name_for_human': schema['summary'],
                'description': schema['description'],
                'parameters': parameters,
                'required': [key for key, value in parameters.items() if 'default' not in value],
                'url': urljoin(openapi_json_url, path)
            })

        return api_dicts


    @classmethod
    def map_path(cls, obj, *paths):
        for path in paths:
            obj = obj.get(path, {})

        return obj

    @classmethod
    def extract_parameters(cls, schema: dict):
        result = {}

        parameters = cls.map_path(schema, 'requestBody', 'content', 'application/json', 'schema', 'properties')
        for name, value in parameters.items():
            meta = {}
            if 'description' in value:
                meta['description'] = value['description']

            if 'type' in value:
                meta['type'] = value['type']

            if 'default' in value:
                meta['default'] = value['default']

            result[name] = meta

        return result


if __name__ == '__main__':
    apibank = ApiBank()
    response = apibank.call(params={'query': '分析东方国信的财务报表'})
    print(response)