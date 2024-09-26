import json
import os
import pathlib
from importlib import import_module
from typing import Dict, Iterator, List, Optional, Union

import json5

from qwen_agent import Agent
from qwen_agent.agents.assistant import KNOWLEDGE_SNIPPET, KNOWLEDGE_TEMPLATE
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, USER, Message
from qwen_agent.log import logger
from qwen_agent.settings import (DEFAULT_MAX_REF_TOKEN, DEFAULT_PARSER_PAGE_SIZE, DEFAULT_RAG_KEYGEN_STRATEGY,
                                 DEFAULT_RAG_SEARCHERS)
from qwen_agent.tools import BaseTool, SimpleDocParser
from qwen_agent.tools.simple_doc_parser import PARSER_SUPPORTED_FILE_TYPES
from qwen_agent.utils.tokenization_qwen import count_tokens
from qwen_agent.utils.utils import extract_files_from_messages, extract_text_from_message, get_file_type, \
    print_traceback


class FullTextMemory(Agent):
    """Memory is special agent for file management.

    By default, this memory can use retrieval tool for RAG.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 files: Optional[List[str]] = None,
                 rag_cfg: Optional[Dict] = None,
                 **kwargs):
        """Initialization the memory.

        Args:
            rag_cfg: The config for RAG. One example is:
              {
                'max_ref_token': 4000,
                'parser_page_size': 500,
                'rag_keygen_strategy': 'SplitQueryThenGenKeyword',
                'rag_searchers': ['keyword_search', 'front_page_search']
              }
              And the above is the default settings.
        """
        self.cfg = rag_cfg or {}

        if 'max_ref_token' not in self.cfg:
            if llm.model in ['qwen2-72b-instruct', 'qwen2.5-72b-instruct']:
                self.cfg['max_ref_token'] = 20000

        self.max_ref_token: int = self.cfg.get('max_ref_token', DEFAULT_MAX_REF_TOKEN)

        cache_dir = os.path.join(pathlib.Path.home(), '.cache', 'llm')
        if isinstance(llm, Dict):
            llm['cache_dir'] = cache_dir
        elif isinstance(llm, BaseChatModel):
            try:
                import diskcache
            except ImportError:
                print_traceback(is_error=False)
                logger.warning(
                    'Caching disabled because diskcache is not installed. Please `pip install diskcache`.')
                cache_dir = None
            os.makedirs(cache_dir, exist_ok=True)
            llm.cache = diskcache.Cache(directory=cache_dir)

        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message)

        self.system_files = files or []
        self.cached_records = []

    def _run(self, messages: List[Message], lang: str = 'en', prompt: str = '', **kwargs) -> Iterator[List[Message]]:
        """This agent is responsible for processing the input files in the message.

         This method stores the files in the knowledge base, and retrievals the relevant parts
         based on the query and returning them.
         The currently supported file types include: .pdf, .docx, .pptx, .txt, .csv, .tsv, .xlsx, .xls and html.

         Args:
             messages: A list of messages.
             lang: Language.

        Yields:
            The message of retrieved documents.
        """
        # process files in messages
        if not self.cached_records:
            rag_files = self.get_rag_files(messages)
            if not rag_files:
                yield [Message(role=ASSISTANT, content='', name='memory')]
                return

            doc_parser = SimpleDocParser(cfg={
                'structured_doc': False
            })

            for file in rag_files:
                record = doc_parser.call(params={'url': file}, **kwargs)
                self.cached_records.append((file, record))

        prompt_template = "{content}\n\n-------------------------\n{prompt}"

        snippets = []
        for file, record in self.cached_records:
            logger.info(f"full text memory processing {file}, prompt: {prompt}")
            *_, last = super()._call_llm([Message(USER, prompt_template.format(prompt=prompt, content=record))])

            snippet = KNOWLEDGE_SNIPPET[lang].format(source=file, content=last[0].content)
            snippets.append(snippet)

        knowledge = KNOWLEDGE_TEMPLATE[lang].format(knowledge="\n\n".join(snippets))
        yield [Message(role=ASSISTANT, content=knowledge, name='memory')]

    def get_rag_files(self, messages: List[Message]):
        session_files = extract_files_from_messages(messages, include_images=False)
        files = self.system_files + session_files
        rag_files = []
        for file in files:
            f_type = get_file_type(file)
            if f_type in PARSER_SUPPORTED_FILE_TYPES and file not in rag_files:
                rag_files.append(file)
        return rag_files
