# import re
# from typing import Iterator, List
#
# import json5
#
# from qwen_agent import Agent
# from qwen_agent.agents.assistant import Assistant
# from qwen_agent.agents.writing import ExpandWriting, OutlineWriting
# from qwen_agent.llm.schema import ASSISTANT, CONTENT, USER, Message
#
# default_plan = """{"action1": "summarize", "action2": "outline", "action3": "expand"}"""
#
#
# def is_roman_numeral(s):
#     pattern = r'^(I|V|X|L|C|D|M)+'
#     match = re.match(pattern, s)
#     return match is not None
#
#
# class WriteFromScratch(Agent):
#
#     def _run(self, messages: List[Message], knowledge: str = '', lang: str = 'en') -> Iterator[List[Message]]:
#
#         response = [Message(ASSISTANT, f'>\n> Use Default plans: \n{default_plan}')]
#         yield response
#         res_plans = json5.loads(default_plan)
#
#         summ = ''
#         outline = ''
#         for plan_id in sorted(res_plans.keys()):
#             plan = res_plans[plan_id]
#             if plan == 'summarize':
#                 summ = ''
#                 # response.append(Message(ASSISTANT, '>\n> Summarize Browse Content: \n'))
#                 # yield response
#                 #
#                 # if not knowledge:
#                 #     if lang == 'zh':
#                 #         user_request = '总结参考资料的主要内容'
#                 #     elif lang == 'en':
#                 #         user_request = 'Summarize the main content of reference materials.'
#                 #     else:
#                 #         raise NotImplementedError
#                 #     sum_agent = Assistant(llm=self.llm)
#                 #     res_sum = sum_agent.run(messages=[Message(USER, user_request)], knowledge=knowledge, lang=lang)
#                 #     chunk = None
#                 #     for chunk in res_sum:
#                 #         yield response + chunk
#                 #     if chunk:
#                 #         response.extend(chunk)
#                 #         summ = chunk[-1][CONTENT]
#                 # else:
#                 #     summ = knowledge
#             elif plan == 'outline':
#                 outline = """"
# 一、引言
# #（一）儿童卒中的定义与分类
# 1. 标准定义及与成人的异同
# 2. 主要分类方式（缺血性、出血性、特殊类型的卒中）
# #（二）研究背景与意义
# 1. 发病率与危害
# 2. 研究现状与不足
# 二、儿童卒中的流行病学
# #（一）发病率与患病率
# 1. 不同年龄段（新生儿、婴儿、儿童等）
# 2. 地区差异（发达国家与发展中国家）
# 3. 性别差异
# #二）病因及相关因素
# 1. 缺血性卒中病因
# #1）先天性心脏病
# 关联机制
# 风险程度
# 研究证据
# 与基因相关的研究
# #2）血管病变：FCA、moyamoya 等
# 特点与分类
# 发病机制与进展
# 风险程度与因素
# 与基因相关的研究
# #3）血液系统异常
# 血栓形成倾向（遗传性与获得性）
# 常见凝血因子异常及相关基因变异
# 与儿童 AIS 的关联及风险评估
# #4）感染与炎症
# 感染在儿童卒中发病中的作用
# 炎症性血管病变的特点与诊断
# 感染与炎症相关研究证据
# 与基因相关的研究
# #5）其他因素
# 急性、慢性系统性疾病等研究证据
# 头颈部外伤
# 与基因相关的研究
# 2. 出血性卒中病因
# #1）血管结构病变
# ①脑动静脉畸形②海绵状血管瘤③脑动静脉瘘④动脉瘤
# 发病的风险程度与因素
# 病变特点与相关研究证据
# 与基因相关的研究
# #2）脑肿瘤：风险因素和病变特点，以及研究证据，与基因相关的研究
# #3）血液病：遗传因素包括血友病、血管性血友病等；获得性因素如特发性血小板减少
# 性紫癜等。
# 病理生理与卒中风险
# 研究证据以及基因相关研究
# #4) 其他风险因素: 包括高血压、感染、药物使用（如 L-天冬酰胺酶治疗）、代谢异常等。
# 发病机制与卒中风险
# 存在的基因研究
# 3. 特殊类型儿童卒中的病因
# #1) 脑静脉窦血栓形成
# 临床表现与诊断挑战
# 病因与危险因素（感染、脱水、血液系统疾病、自身免疫及代谢疾病等）
# 相关的基因研究证据
# 治疗策略（抗凝治疗、支持治疗、介入治疗等）与疗效评估
# #2) 镰状细胞病相关卒中
# 病理生理机制与风险因素
# 遗传性基因相关研究
# 预防策略（如输血治疗、羟基脲的应用）
# 三、儿童卒中的临床表现与诊断
# #（一）临床症状与体征
# 1. 不同类型卒中的常见表现（如偏瘫、失语、头痛、抽搐等）
# 2. 年龄相关的症状特点（新生儿与儿童的差异）
# 3. 与成人的对比
# （二）诊断方法与技术
# #1. 神经影像学
# MRI与MRA的应用（包括序列选择与影像特征）
# CT与CTA的优势与局限性
# 血管成像技术在动脉病变诊断中的价值（如DSA等）
# #2. 实验室检查
# 血液学检查（凝血功能、血小板、血脂等）
# 脑脊液检查（在特定情况下的应用及意义）
# 基因检测（常见基因检测项目）
# #3. 诊断流程与挑战
# 快速准确诊断的难点与应对策略
# 误诊与漏诊情况分析及改进措施
# 四、儿童卒中的治疗与管理
# （一）急性治疗
# #1. 超急性卒中疗法
# 风险评估 (pedNIHSS)
# 溶栓治疗（药物选择、剂量、时间窗等）
# 儿童与成人的差异及研究现状
# 机械取栓术，技术应用的可行性与安全性
# RCT研究疗效分析，临床试验进展
# #2. 一般支持治疗
# 血压、血糖、体温管理的策略与目标
# 癫痫发作的控制措施
# 激素治疗研究意义
# 颅内压增高的处理（如脱水治疗、脑脊液引流等）
# #3. 手术干预
# 颅内血管病变的手术治疗（如moyamoya病的血管重建术）
# 手术方式的选择与效果评估
# 围手术期管理与并发症预防
# 其他手术治疗（如脑血肿清除术、颅骨减压术等）的适应证与疗效
# （二）二级预防
# #1. 抗血栓治疗
# 抗血小板药物（阿司匹林等）的使用与疗效
# 抗凝药物的选择与监测
# 不同病因下的治疗策略（如心脏源性、血管病变等）
# 治疗时间与疗程的确定
# #2. 康复治疗
# 康复介入的时机与重要性
# 康复方法与技术（如物理治疗、作业治疗、言语治疗等）
# 新兴康复手段的研究与应用
# 五、儿童卒中的预后
# #（一）预后评估指标
# 1. 死亡率与生存率
# 2. 神经功能恢复（运动、认知、语言等方面）
# 3. 生活质量评估
# #（二）影响预后的因素
# 1. 卒中类型与严重程度
# 2. 治疗及时性与有效性
# 3. 合并症与并发症
# 4. 康复治疗的参与度与质量
# 六、结论与展望
# #（一）研究总结
# 1. 儿童卒中的主要研究成果与共识
# 2. 现有研究的局限性与不足之处
# #（二）未来研究方向
# 1. 基础研究（如发病机制、基因调控等）
# 2. 临床研究（治疗方法优化、新药物研发、康复技术创新等）
# 3. 多学科合作与综合管理模式的探索
# 4. 提高公众认知与早期诊断的策略
# """
#                 # response.append(Message(ASSISTANT, '>\n> Generate Outline: \n'))
#                 # yield response
#                 #
#                 # otl_agent = OutlineWriting(llm=self.llm)
#                 # res_otl = otl_agent.run(messages=messages, knowledge=summ, lang=lang)
#                 # chunk = None
#                 # for chunk in res_otl:
#                 #     yield response + chunk
#                 # if chunk:
#                 #     response.extend(chunk)
#                 #     outline = chunk[-1][CONTENT]
#             elif plan == 'expand':
#                 response.append(Message(ASSISTANT, '>\n> Writing Text: \n'))
#                 yield response
#
#                 outline_list_all = outline.split('\n')
#                 outline_list = []
#                 for x in outline_list_all:
#                     if x.startswith('#'):
#                         outline_list.append(x[1:])
#                     # if is_roman_numeral(x):
#                     #     outline_list.append(x)
#
#                 outline = outline.replace("#", "")
#                 otl_num = len(outline_list)
#                 for i, v in enumerate(outline_list):
#                     response.append(Message(ASSISTANT, '>\n# '))
#                     yield response
#
#                     index = i + 1
#                     capture = v.strip()
#                     capture_later = ''
#                     if i < otl_num - 1:
#                         capture_later = outline_list[i + 1].strip()
#                     exp_agent = ExpandWriting(llm=self.llm)
#                     res_exp = exp_agent.run(
#                         messages=messages,
#                         knowledge=knowledge,
#                         outline=outline,
#                         index=str(index),
#                         capture=capture,
#                         capture_later=capture_later,
#                         lang=lang,
#                     )
#                     chunk = None
#                     for chunk in res_exp:
#                         yield response + chunk
#                     if chunk:
#                         response.extend(chunk)
#             else:
#                 pass
import re
from typing import Iterator, List

import json5

from qwen_agent import Agent
from qwen_agent.agents.assistant import Assistant
from qwen_agent.agents.writing import ExpandWriting, OutlineWriting
from qwen_agent.llm.schema import ASSISTANT, CONTENT, USER, Message

default_plan = """{"action1": "summarize", "action2": "outline", "action3": "expand"}"""


def is_roman_numeral(s):
    pattern = r'^(I|V|X|L|C|D|M)+'
    match = re.match(pattern, s)
    return match is not None


class WriteFromScratch(Agent):

    def _run(self, messages: List[Message], knowledge: str = '', lang: str = 'en') -> Iterator[List[Message]]:

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
                res_otl = otl_agent.run(messages=messages, knowledge=summ, lang=lang)
                chunk = None
                for chunk in res_otl:
                    yield response + chunk
                if chunk:
                    response.extend(chunk)
                    outline = chunk[-1][CONTENT]
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