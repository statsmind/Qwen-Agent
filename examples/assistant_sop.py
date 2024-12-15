import json
import os
import re
from collections import namedtuple
from typing import List, Dict, Tuple

import docx

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import Message, USER, ASSISTANT
from markdowntodocx.markdownconverter import markdownToWordFromString
# from Markdown2docx import Markdown2docx

CONTEXT_TEMPLATE = """#-*- TASK -*-

为了标准化、规范化地开展临床试验实施和试验设计、数据管理、统计分析和临床评价工作，加强临床试验质量控制，在参考ICH、FDA和NMPA等机构法规和指南的基础上，我公司需要制定一系列标准操作规程（SOP），内容涵盖临床试验设计、测量和评价的主要关键环节，适用于以下人员：机构研究者、数据管理人员、数据处理人员、监查员、生物统计师、法规事务人员和所有临床研究专业人员。你的任务是根据我的指令一步步完成这些文档。

文档列表：
{sop_structure}


#-*- RULES -*-

你的任务是按照我的指令生成这些文档。文档格式使用markdown，后面我会转化成docx文档。文档内容要详细，不要简单几句话就敷衍我，
如果文档有附件，请务必补充上附件的详细内容。


#-*- INSTRUCT -*-

"""

DOC_GEN_TEMPLATE = """帮我完成文档《{doc_name}》，开始："""

DOC_REFINE_TEMPLATE = """不错，你生成了文档的基础框架，但是存在以下问题：

1. 很多内容都是一笔带过，没有详细展开
2. 不要使用嵌套的 "```" 标记，这会给docx文档生成带来麻烦

请重新生成文档："""


class SopDoc:
    sop_storage_path = './workspace/sop_documents'

    def __init__(self, section_name: str, doc_name: str, sop_structure: Dict[str, List[str]]):
        self.section_name = section_name
        self.doc_name = doc_name
        self.sop_structure = sop_structure
        # self.content = ""
        # self.refined_content = ""
        self.prev = None
        self.next = None
        self.props = {}

        # self.load()

    @property
    def raw_md_path(self) -> str:
        raw_md_path = os.path.join(self.sop_storage_path, self.section_name, self.doc_name + ".txt")
        os.makedirs(os.path.dirname(raw_md_path), exist_ok=True)
        return raw_md_path

    @property
    def refined_md_path(self) -> str:
        refined_md_path = self.raw_md_path.replace("sop_documents", "sop_documents_refine")
        os.makedirs(os.path.dirname(refined_md_path), exist_ok=True)
        return refined_md_path

    @property
    def docx_path(self) -> str:
        docx_path = self.raw_md_path.replace("sop_documents", "sop_documents_docx")
        docx_path = docx_path.replace(".txt", ".docx")
        os.makedirs(os.path.dirname(docx_path), exist_ok=True)
        return docx_path

    def __repr__(self):
        return self.section_name + "/" + self.doc_name

    def get_markdown(self, force_update: bool = False) -> str:
        markdown = self.props.get("markdown", "")
        if not markdown:
            raw_md_path = self.raw_md_path
            if os.path.exists(raw_md_path):
                with open(raw_md_path, "r", encoding="UTF-8") as fp:
                    markdown = fp.read()
                    self.props['markdown'] = markdown

        if markdown and not force_update:
            return markdown



    def load(self) -> Tuple[str, str]:
        doc_path = self.raw_md_path
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="UTF-8") as fp:
                self.content = fp.read()

        doc_refine_path = self.refined_md_path
        if os.path.exists(doc_refine_path):
            with open(doc_refine_path, "r", encoding="UTF-8") as fp:
                self.refined_content = fp.read()

        return self.content, self.refined_content

    def save(self):
        doc_path = self.raw_md_path
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)

        if self.content:
            with open(doc_path, "w", encoding="UTF-8") as fp:
                fp.write(self.content)

    def save_refine(self):
        doc_refine_path = self.refined_md_path

        if self.refined_content:
            with open(doc_refine_path, "w", encoding="UTF-8") as fp:
                fp.write(self.refined_content)

    def save_docx(self):
        doc_path = self.raw_md_path
        content = self.content

        if self.refined_content:
            doc_path = self.refined_md_path
            content = self.refined_content

        if not content or not os.path.exists(doc_path):
            return

        docx_path = self.docx_path
        if os.path.exists(docx_path) and os.path.getmtime(docx_path) > os.path.getmtime(doc_path):
            return

        markdownToWordFromString(content, docx_path)
        # content = content.replace("**", "")
        # project = Markdown2docx(self.doc_name, doc_path, docx_path, markdown=content)
        # if self.doc_name not in self.content.split("\n")[0]:
        #     project.doc.add_heading(self.doc_name, level=0)
        # project.eat_soup()
        # project.save()

    @classmethod
    def flatten_sop_structure(cls, sop_structure):
        items = []
        for section_name, doc_names in sop_structure.items():
            items.append(section_name)
            items.extend([f"《{doc_name}》" for doc_name in doc_names])
        return "\n".join(items)

    def pipeline(self, agent):
        self.gen_content(agent)
        self.refine()
        self.save_docx()

    def gen_content(self, agent: Assistant):
        if self.content and not force:
            return

        sop_doc_ref = self
        sop_doc_chain = [sop_doc_ref]
        while len(sop_doc_chain) < 4 and sop_doc_ref.prev is not None:
            sop_doc_chain.append(sop_doc_ref.prev)
            sop_doc_ref = sop_doc_ref.prev
        sop_doc_chain = sop_doc_chain[::-1]

        messages = []
        context_prompt = CONTEXT_TEMPLATE.format(sop_structure=self.flatten_sop_structure(self.sop_structure))

        chain_size = len(sop_doc_chain)
        for idx in range(chain_size):
            doc_prompt = DOC_GEN_TEMPLATE.format(doc_name=sop_doc_chain[idx].doc_name)
            if idx == 0:
                messages.append(Message(role=USER, content=context_prompt + doc_prompt))
            else:
                messages.append(Message(role=USER, content=doc_prompt))

            if idx < chain_size - 1:
                messages.append(Message(role=ASSISTANT, content=sop_doc_chain[idx].content))

        *_, last = agent.run(messages)
        self.content = last[0].content

        self.save()

    def refine(self):
        doc_refine_path = self.refined_md_path

        if not os.path.exists(doc_refine_path):
            print(f"Refining {self.doc_name}")
            messages = []
            context_prompt = CONTEXT_TEMPLATE.format(sop_structure=self.flatten_sop_structure(self.sop_structure))
            doc_prompt = DOC_GEN_TEMPLATE.format(doc_name=self.doc_name)

            messages.append(Message(role=USER, content=context_prompt + doc_prompt))
            messages.append(Message(role=ASSISTANT, content=self.content))
            messages.append(Message(role=USER, content=DOC_REFINE_TEMPLATE))

            *_, last = agent.run(messages)
            self.refined_content = last[0].content
            self.save_refine()
        else:
            with open(doc_refine_path, "r", encoding='UTF-8') as fp:
                self.refined_content = fp.read()

        # os.makedirs(os.path.dirname(docx_refine_path), exist_ok=True)
        #
        # project = Markdown2docx(self.doc_name, doc_refine_path, docx_refine_path)
        # if self.doc_name not in refined_content.split("\n")[0]:
        #     project.doc.add_heading(self.doc_name, level=0)
        # project.eat_soup()
        # project.save()


sop_structure: Dict[str, List[str]] = {
    "1.SOP管理": [
        "关于标准操作规程文件的保密声明",
        "标准操作规程文件的审阅及批准",
        "标准操作规程文件的维护"
    ],
    "2.临床监查": [
        "研究中心启动及关闭报告",
        "研究中心依从性不佳解决计划",
        "临床研究人员授权及签名表",
        "受试者知情同意书",
        "研究者简历",
        "研究者手册",
        "临床试验用产品接收单",
        "临床试验用药品发放返还表",
        "受试者筛选与入选表",
        "受试者身份识别表",
        "原始数据核查计划",
        "临床研究常规监查报告"
    ],
    "3.培训": [
        "研究中心培训记录表",
        "员工项目培训记录表"
    ],
    "4.QA": [
        "质量管理体系手册",
        "质量保证稽查计划",
        "供应商管理"
    ],
    "5.QM": [
        "研究中心质量审核访视规程",
        "整改预防措施计划",
        "投诉管理及跟进"
    ],
    "6.SSU": [
        "临床项目启动",
        "项目可行性调研"
    ],
    "7.记录管理": [
        "电子记录存档",
        "文件存档",
        "记录管理计划"
    ],
    "8.临床安全管理": [
        "临床试验案例管理",
        "临床试验安全报告"
    ],
    "9.项目管理": [
        "项目的启动和计划",
        "项目的执行和控制",
        "项目关闭",
        "项日管理计划"
    ],
    "10.注册事务": [
        "医疗器械临床试验备案",
        "医疗器械注册程序",
        "药品注册程序",
        "药物临床试验实施前备案",
        "人类遗传资源采集、收集、买卖、出口、出境审批申请",
        "进囗药品注册检验"
    ],
    "11.生物统计": [
        "生物统计原则",
        "生物统计编程质量管理规范",
        "QC和高级生物统计审阅计划",
        "统计分析计划",
        "揭盲计划模板",
        "统计分析报告模板",
        "SAS编程规范"
    ],
    "12.数据管理": [
        "CRF的设计",
        "电子数据传输表",
        "账户管理安全性与权限控制",
        "数据录入",
        "数据传输的处理",
        "数据管理编程规范",
        "数据管理的质量保证",
        "研究数据文档的控制与保存",
        "数据库建立、锁定、迁移",
        "偏离的管理",
        "数据追踪指南",
        "数据清理（EDC项目）",
        "EDC项目的质量控制计划"
    ],
    "13.研究中心管理服务": [
        "项目文件管理及保存",
        "临床研究协调员及项目经理工作范围",
        "合同签署、变更",
        "业务连续性、灾难恢复计划",
        "财务管理"
    ],
    "14.医学科学事务": [
        "临床评价报告模板",
        "方案的医学审阅",
        "临床研究报告的撰写及医学审阅",
        "医学监查计划",
        "研究者手册",
        "医学编码列表",
        "医学数据审阅",
        "方案偏离审阅",
    ]
}

sop_section_names = list(sop_structure.keys())

sop_docs = []
for section_name in sop_section_names:
    for doc_name in sop_structure[section_name]:
        sop_docs.append(SopDoc(section_name=section_name, doc_name=doc_name, sop_structure=sop_structure))

sop_doc_num = len(sop_docs)

for idx in range(sop_doc_num):
    if idx < sop_doc_num - 1:
        sop_docs[idx].next = sop_docs[idx + 1]

    if idx > 0:
        sop_docs[idx].prev = sop_docs[idx - 1]


if __name__ == '__main__':
    agent = Assistant(
        llm={'model': 'qwen-max'},
        name='sop_assistant',
        description='SOP文档助手',
        system_message="你是一位资深的临床试验质量控制专家，熟悉ICH、FDA和NMPA等机构法规和指南，为我公司制定一系列标准操作规程（SOP）。"
    )

    for sop_doc_idx, sop_doc in enumerate(sop_docs):
        sop_doc.gen_content(agent)
        sop_doc.refine()
        sop_doc.save_docx()