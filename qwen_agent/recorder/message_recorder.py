import json
import os.path
import pathlib
import uuid
from typing import List, Literal, Union, Dict
from markdown import markdown
from qwen_agent.llm.schema import Message, USER, ASSISTANT
from qwen_agent.log import logger


class MessageRecorder(object):
    def __init__(self, output_formats: List[Literal['jsonl', 'html']] = None):
        if output_formats is None:
            output_formats = []

        self.output_formats = output_formats
        self.jsonl_fp = None
        self.html_fp = None
        self.step_index = -1

    def _reset(self):
        if self.jsonl_fp is not None:
            self.jsonl_fp.close()
            self.jsonl_fp = None

        if self.html_fp is not None:
            self.html_fp.close()
            self.html_fp = None

        if len(self.output_formats) > 0:
            session_id = uuid.uuid4().hex
            base_path = os.path.join(pathlib.Path.home(), f".qwen_agent/{session_id}")
            os.makedirs(os.path.dirname(base_path), exist_ok=True)

            if 'jsonl' in self.output_formats:
                self.jsonl_fp = open(base_path + ".jsonl", "w", encoding="utf-8")
                logger.info(f"Writing jsonl file {base_path + '.jsonl'}")
            if 'html' in self.output_formats:
                self.html_fp = open(base_path + ".html", "w", encoding="utf-8")
                logger.info(f"Writing html file {base_path + '.html'}")

                css_content = open(os.path.join(os.path.dirname(__file__), 'assets/style.css'), 'r', encoding="utf-8").read()
                self.html_fp.write(f"""<html>\n<head><meta charSet="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, minimum-scale=1.0"/><meta http-equiv="X-UA-Compatible" content="ie=edge,chrome=1"/><meta http-equiv="Cache-Control" content="no-siteapp"/><meta http-equiv="Cache-Control" content="no-transform"/>\n<style>\n{css_content}\n</style></head>\n<body>\n""")

    def start(self, message: Union[Dict, Message]):
        self._reset()
        self._dump_message(message)
        self.step_index = -1

    def step(self, messages: List[Union[Dict, Message]], is_final: bool = False):
        if is_final:
            new_messages = messages + [Message(ASSISTANT, '')]
        else:
            new_messages = messages[:len(messages)-1]
        for index, rsp in enumerate(new_messages):
            if index > self.step_index:
                self._dump_message(rsp)
                self.step_index = index

        if is_final:
            if self.jsonl_fp is not None:
                self.jsonl_fp.close()
                self.jsonl_fp = None

            if self.html_fp is not None:
                self.html_fp.close()
                self.html_fp = None

    def _dump_message(self, message: Union[Dict, Message]):
        if isinstance(message, dict):
            message = Message(**message)

        if message.role not in ['user', 'assistant']:
            return

        if self.jsonl_fp is not None:
            self.jsonl_fp.write(json.dumps(message.model_dump(), ensure_ascii=False) + "\n")
            self.jsonl_fp.flush()

        if self.html_fp is not None:
            content = []
            if isinstance(message.content, list):
                for item in message.content:
                    if item.text:
                        content.append(item.text)
            else:
                content = [message.content]
            content = "\n".join(content)
            content = markdown(content).strip()

            if len(content) > 0:
                snippet = f"<div class='role-{message.role}'>{content}</div>"
                self.html_fp.write(snippet + "\n")
                self.html_fp.flush()

if __name__ == '__main__':
    a = []
    a = a[:len(a)-1]
    print(a)