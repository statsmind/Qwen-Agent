import json
from typing import List, Literal, Iterator

import numpy as np
from pypinyin import lazy_pinyin, Style

from qwen_agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, USER

ALL_HANS = [chr(j) for j in range(0x4e00, 0x9fa5 + 1)]

ALL_LYRICS = []
with open(r'F:\resources\data\chinese-lyrics\music.json', 'r', encoding='utf-8') as f:
    for line in f:
        ALL_LYRICS.append(json.loads(line))


def find_geci(end: str):
    result = []

    for lyric in ALL_LYRICS:
        for geci in lyric['geci']:
            if geci.endswith(end):
                result.append(geci)

    return np.unique(result)


def get_last_hanz(text: str) -> str:
    for i in range(0, len(text)):
        hanz = text[-(i+1)]
        if hanz in ALL_HANS:
            return hanz


def get_yunmu(hanz: str):
    yunmu = lazy_pinyin(hanz, style=Style.FINALS)
    return yunmu[0]


def find_all_by_yunmu(yunmu: str):
    hanzs = []

    for hanz in ALL_HANS:
        if get_yunmu(hanz) == yunmu:
            hanzs.append(hanz)

    return hanzs


# 歌词创作
class WriteLyric(Assistant):
    def _run(self,
             messages: List[Message],
             lang: Literal['en', 'zh'] = 'zh',
             knowledge: str = '',
             **kwargs) -> Iterator[List[Message]]:
        query = messages[-1].text_content()
        last_hanz = get_last_hanz(query)
        last_yunmu = get_yunmu(last_hanz)
        end_hanzs = find_all_by_yunmu(last_yunmu)

        for end_hanz in end_hanzs:
            kb_gecis = find_geci(end_hanz)
            if len(kb_gecis) == 0:
                continue

            kb_gecis = "\n".join(kb_gecis[:300])
            user_message = Message(USER, """你是歌词创作大师，你的任务是帮我创造歌词.

下面这些歌词可以参考：
{kb_gecis}
            
我有一句种子歌词：{seed_lyric}
根据种子歌词的场景和心情，帮我创作30句独立的多样化的歌词, 歌词要以汉字"{hanz}"结尾。整体作词风格慵懒。

输出格式：
1. 歌词
歌词的意境
2. 歌词
歌词的意境
...
""".format(kb_gecis=kb_gecis, seed_lyric=query, hanz=end_hanz))
            *_, last = super()._run([user_message])
            print(last)
            break


if __name__ == '__main__':
    write_lyric = WriteLyric(llm={"generate_cfg": {"max_input_tokens": 31000}}, name='助理', description='我的私人助理', system_message="你是歌词创作大师，你的任务是帮我创造歌词", auto_tools=False)
    message = Message(USER, "我只能在你不注意的时候亲吻你的脸颊。")
    write_lyric.run_nonstream(messages=[message])
