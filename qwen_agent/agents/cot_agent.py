import copy
import json
from typing import List, Literal, Iterator

from qwen_agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, SYSTEM, ASSISTANT, USER, FUNCTION
from qwen_agent.utils.utils import has_chinese_chars

COT_TEMPLATE_EN = """You are an advanced AI reasoning assistant tasked with delivering a comprehensive analysis of a specific problem or question.  Your goal is to outline your reasoning process in a structured and transparent manner, with each step reflecting a thorough examination of the issue at hand, culminating in a well-reasoned conclusion.

### Structure for Each Reasoning Step:
1.  **Title**: Clearly label the phase of reasoning you are currently in.
2.  **Content**: Provide a detailed account of your thought process, explaining your rationale and the steps taken to arrive at your conclusions.
3.  **Next Action**: Decide whether to continue with further reasoning or if you are ready to provide a final answer.

### Response Format:
Please return the results in the following JSON format:
- `title`: A brief label for the current reasoning phase.
- `content`: An in-depth explanation of your reasoning process for this step.
- `next_action`: Choose `'continue'` to proceed with further reasoning or `'final_answer'` to conclude.

### Key Instructions:
1.  Conduct **at least 5 distinct reasoning steps**, each building on the previous one.
2.  **Acknowledge the limitations** inherent to AI, specifically what you can accurately assess and what you may struggle with.
3.  **Adopt multiple reasoning frameworks** to resolve the problem or derive conclusions, such as:
- **Deductive reasoning** (drawing specific conclusions from general principles)
- **Inductive reasoning** (deriving broader generalizations from specific observations)
- **Abductive reasoning** (choosing the best possible explanation for the given evidence)
- **Analogical reasoning** (solving problems through comparisons and analogies)
4.  **Critically analyze your reasoning** to identify potential flaws, biases, or gaps in logic.
5.  When reviewing, apply a **fundamentally different perspective or approach** to enhance your analysis.
6.  **Employ at least 2 distinct reasoning methods** to derive or verify the accuracy of your conclusions.
7.  **Incorporate relevant domain knowledge** and **best practices** where applicable, ensuring your reasoning aligns with established standards.
8.  **Quantify certainty levels** for each step and your final conclusion, where applicable.
9.  Consider potential **edge cases or exceptions** that could impact the outcome of your reasoning.
10.  Provide **clear justifications** for dismissing alternative hypotheses or solutions that arise during your analysis.

### Example JSON Output:

```json
{
"title": "Initial Problem Analysis",
"content": "To approach this problem effectively, I'll first break down the given information into key components.  This involves identifying... [detailed explanation]...  By structuring the problem in this way, we can systematically address each aspect.",
"next_action": "continue"
}
```
"""

COT_TEMPLATE_ZH = """您是一名高级 AI 推理助手，负责对特定问题或疑问进行全面分析。 你的目标是以结构化和透明的方式概述你的推理过程，每一步都反映了对手头问题的彻底审查，最终得出一个合理的结论。

### 每个推理步骤的结构：
1. **Title**：清楚地标明你目前所处的推理阶段。
2. **Content**：详细描述您的思考过程，解释您的理由以及得出结论所采取的步骤。
3. **Next Action**：决定是继续进一步推理，还是准备好提供最终答案。
4. **Placeholder**: 占位符，始终等于空


### 关键说明：
1. 执行 **至少 5 个不同的推理步骤**，每个步骤都建立在前一个步骤的基础上。
2. **承认 AI 固有的局限性**，特别是您可以准确评估的内容和您可能难以解决的问题。
3. **采用多种推理框架** 来解决问题或得出结论，例如：
- **演绎推理**（从一般原则中得出具体结论）
- **归纳推理**（从具体观察中得出更广泛的概括）
- **归纳推理**（为给定的证据选择最佳解释）
- **类比推理**（通过比较和类比解决问题）
4. **批判性地分析你的推理**，以识别逻辑中的潜在缺陷、偏见或差距。
5. 审查时，应用 **完全不同的观点或方法** 来增强您的分析。
6. **采用至少 2 种不同的推理方法**来推导出或验证您的结论的准确性。
7. 在适用的情况下，**结合相关的领域知识和**最佳实践**，确保您的推理符合既定标准。
8. **量化每个步骤的确定性水平**和您的最终结论（如适用）。
9. 考虑可能影响您推理结果的潜在 **边缘情况或异常** 。
10. 提供 **明确的理由** 以驳斥分析过程中出现的替代假设或解决方案。


### 输出格式：
请以以下 JSON 格式返回结果：
- 'title'：当前推理阶段的简短标签。
- 'content'：深入说明此步骤的推理过程。
- 'next_action'： 选择 `'continue'` 继续进一步推理，或者选择 `'final_answer'` 得出结论。
- 'placeholder': 始终为空

### JSON 输出示例：

```json
{
"title": "初始问题分析",
"content": "为了有效地解决这个问题，我首先将给定的信息分解为关键部分。 这包括识别...[详细说明]... 通过以这种方式构建问题，我们可以系统地解决每个方面",
"next_action": "continue"，
"placeholder": ""
}
```
"""

COT_TEMPLATE = {'zh': COT_TEMPLATE_ZH, 'en': COT_TEMPLATE_EN}

ASSISTANT_STEP_ZH = "谢谢！现在，我将按照我的指示逐步思考，在分解问题后从头开始。"
ASSISTANT_STEP_EN = "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."

ASSISTANT_STEP = {'zh': ASSISTANT_STEP_ZH, 'en': ASSISTANT_STEP_EN}


class COTAgent(Assistant):
    def _run(self,
             messages: List[Message],
             lang: Literal['en', 'zh', 'auto'] = 'auto',
             knowledge: str = '',
             **kwargs) -> Iterator[List[Message]]:

        if lang == 'auto':
            if has_chinese_chars(messages[-1].content):
                lang = 'zh'
            else:
                lang = 'en'

        system_prompt = COT_TEMPLATE[lang]

        new_messages = copy.deepcopy(messages)
        if new_messages[0].role == SYSTEM:
            new_messages[0].content = system_prompt
        else:
            new_messages = [Message(SYSTEM, content=system_prompt)] + new_messages
        # new_messages.append(Message(FUNCTION, ASSISTANT_STEP[lang]))

        response = []

        for _ in range(50):
            output_stream = super()._run(new_messages, lang, knowledge, stop=['"placeholder":'], **kwargs)

            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output

            if len(output) == 0:
                break

            response.extend(output)

            if output[-1].content.strip().endswith(","):
                output[-1].content += '"placeholder":""}```\n'

            assistant_reply = output[-1].content
            last_snippet = assistant_reply.split("```json\n")[-1]
            if "```\n" not in last_snippet:
                new_messages.extend(output)
                continue

            last_block = last_snippet.split("```\n")[0]

            last = json.loads(last_block)
            if last['next_action'] != 'continue':
                break

            new_messages.extend([
                Message(ASSISTANT, output[-1].content),
                Message(USER, 'continue'),
            ])

        yield response
