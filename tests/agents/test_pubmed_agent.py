from qwen_agent.agents.medical.pubmed_agent import PubMedAgent
from qwen_agent.llm.schema import Message


def test_pubmed_agent():
    pubmed_agent = PubMedAgent()
    *_, response = pubmed_agent.run(messages=[Message('user', '安宫牛黄丸动物或临床研究的证据')])
    print(response)
