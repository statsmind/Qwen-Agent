import re
from typing import Iterator, List, Literal

import json5

from qwen_agent import Agent
from qwen_agent.agents.assistant import Assistant
from qwen_agent.agents.writing import ExpandWriting, OutlineWriting
from qwen_agent.llm.schema import ASSISTANT, CONTENT, USER, Message

default_plan = """{"action1": "summarize", "action2": "outline", "action3": "expand"}"""
default_plan = """{"action2": "outline", "action3": "expand"}"""

DEFAULT_OUTLINE = """
I. 引言
    A. 儿童卒中的定义与分类
    B. 研究背景与重要性

II. 临床诊断与评估工具
    A. PedRACE：儿科卒中快速动脉闭塞评价量表
        1. 设计与验证
        2. 可靠性与准确性
    B. 神经影像学在儿科卒中的应用
        1. MRI血管壁成像
        2. 高级成像技术：ASL和CEST MRI
    C. 血管造影与血管壁成像
        1. 脑血管造影
        2. 脑血管壁成像

III. 病因学与风险因素
    A. 儿童卒中的病因学研究
        1. 感染与疫苗接种
        2. 遗传与环境因素
    B. 风险因素分析
        1. 血压、血糖与体温
        2. 血栓形成倾向与遗传性血栓病
    C. 特殊群体的研究
        1. 新生儿出血性卒中
        2. 早产儿与低出生体重婴儿

IV. 治疗与干预
    A. 急性期管理
        1. 血管内治疗
            a. 机械取栓
            b. 内科治疗
        2. 抗凝与抗血小板治疗
    B. 预防策略
        1. 二级预防
        2. 血管疾病管理
    C. 康复与长期管理
        1. 康复治疗
        2. 心理社会功能评估

V. 预后与长期影响
    A. 认知与神经功能结局
        1. 认知功能障碍
        2. 运动功能障碍
    B. 社会心理适应与生活质量
        1. 家庭负担与社会支持
        2. 学业与职业影响
    C. 长期随访研究
        1. 成年后的健康状况
        2. 神经发育结果

VI. 未来方向与研究挑战
    A. 早期诊断与生物标志物
    B. 个体化治疗与精准医学
    C. 多中心合作与大数据
    D. 预防策略与公共卫生政策

VII. 结论
    A. 儿童卒中研究的最新进展概览
    B. 对未来研究的展望
    C. 对临床实践的影响与建议
    D. 强调国际合作与资源共享的重要性

VIII. 参考文献

"""
def is_roman_numeral(s):
    pattern = r'^(I|V|X)+'
    match = re.match(pattern, s)
    return match is not None


class WriteFromScratch(Agent):

    def _run(self, messages: List[Message], knowledge: str = '', lang: Literal['zh', 'en'] = 'zh') -> Iterator[List[Message]]:
        response = [Message(ASSISTANT, f'>\n> Use Default plans: \n{default_plan}')]
        yield response
        res_plans = json5.loads(default_plan)

        summ = ''
        outline = ''
        for plan_id in sorted(res_plans.keys()):
            plan = res_plans[plan_id]
            if plan == 'summarize':
                response.append(Message(ASSISTANT, '>\n> Summarize Browse Content: \n'))
                yield response

                if lang == 'zh':
                    user_request = '总结参考资料的主要内容'
                elif lang == 'en':
                    user_request = 'Summarize the main content of reference materials.'
                else:
                    raise NotImplementedError
                sum_agent = Assistant(llm=self.llm)
                res_sum = sum_agent.run(messages=[Message(USER, user_request)], knowledge=knowledge, lang=lang)
                chunk = None
                for chunk in res_sum:
                    yield response + chunk
                if chunk:
                    response.extend(chunk)
                    summ = chunk[-1][CONTENT]
            elif plan == 'outline':
                response.append(Message(ASSISTANT, '>\n> Generate Outline: \n'))
                yield response

                otl_agent = OutlineWriting(llm=self.llm)
                res_otl = otl_agent.run(messages=messages, knowledge=knowledge, lang=lang)
                chunk = None
                for chunk in res_otl:
                    yield response + chunk
                if chunk:
                    response.extend(chunk)
                    # chunk[-1][CONTENT] = DEFAULT_OUTLINE
                    outline = chunk[-1][CONTENT]
                    outline += "\nXX. References:"
            elif plan == 'expand':
                response.append(Message(ASSISTANT, '>\n> Writing Text: \n'))
                yield response

                outline_list_all = outline.split('\n')
                outline_list = []
                for x in outline_list_all:
                    if is_roman_numeral(x):
                        outline_list.append(x)

                otl_num = len(outline_list)
                for i, v in enumerate(outline_list):
                    response.append(Message(ASSISTANT, '>\n# '))
                    yield response

                    index = i + 1
                    capture = v.strip()
                    capture_later = ''
                    if i < otl_num - 1:
                        capture_later = outline_list[i + 1].strip()
                    exp_agent = ExpandWriting(llm=self.llm)
                    res_exp = exp_agent.run(
                        messages=messages,
                        knowledge=knowledge,
                        outline=outline,
                        index=str(index),
                        capture=capture,
                        capture_later=capture_later,
                        lang=lang,
                    )
                    chunk = None
                    for chunk in res_exp:
                        yield response + chunk
                    if chunk:
                        response.extend(chunk)
            else:
                pass
