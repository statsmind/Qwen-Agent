from marker.convert import convert_single_pdf

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, USER, ContentItem
from qwen_agent.tools.simple_doc_parser import parse_pdf, get_plain_doc, parse_word, parse_html_bs
import marker


if __name__ == '__main__':
    content = open(r'C:\Users\james\Desktop\儿童卒中登记表.txt', 'r', encoding='utf-8').read()
    variables = open(r'C:\Users\james\Desktop\变量.txt', 'r', encoding='utf-8').read()

    bot = Assistant()
    *_, last = bot.run([Message(USER, f"""你是一位资深的医学专家，请结合儿童卒中的特点及未来方向回答我的问题。

# 儿童卒中登记研究方案
{content}


# 从近几年关于儿童卒中的论文中搜集的研究变量
{variables}


# 任务
请参考上面的近几年关于儿童卒中的论文中搜集的研究变量，儿童卒中登记研究方案缺少哪些变量？""")])
    print(last)
