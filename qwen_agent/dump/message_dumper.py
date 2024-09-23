import json
import os.path
import uuid
from typing import List, Literal, Union, Dict
from markdown import markdown
from qwen_agent.llm.schema import Message, USER


class MessageDumper(object):
    def __init__(self, output_formats: List[Literal['jsonl', 'html']] = None):
        if output_formats is None:
            output_formats = ['jsonl', 'html']

        self.step_messages: List[Union[Dict, Message]] = []

        session_id = uuid.uuid4().hex
        base_path = os.path.join(os.getenv("HOME"), f".qwen_agent/{session_id}")
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        if 'jsonl' in output_formats:
            self.jsonl_fp = open(base_path + ".jsonl", "w", encoding="utf-8")
        else:
            self.jsonl_fp = None

        if 'html' in output_formats:
            self.html_fp = open(base_path + ".html", "w", encoding="utf-8")

            css_content = open(os.path.join(os.path.dirname(__file__), 'assets/style.css'), 'r', encoding="utf-8").read()
            self.html_fp.write(f"<html>\n<head>\n<style>\n{css_content}\n</style></head>\n<body>\n")
        else:
            self.html_fp = None

    def start_loop(self, message: Union[Dict, Message]):
        self._dump_message(message)

    def step(self, messages: List[Union[Dict, Message]]):
        self.step_messages = messages

    def end_loop(self):
        for message in self.step_messages:
            self._dump_message(message)

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


class DummyDumper(object):
    def __init__(self, output_formats: List[Literal['jsonl', 'html']] = None):
        pass

    def start_loop(self, message: Union[Dict, Message]):
        pass

    def step(self, messages: List[Union[Dict, Message]]):
        pass

    def end_loop(self):
        pass
