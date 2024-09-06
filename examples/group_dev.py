"""A group chat gradio demo"""
import json

import json5

from qwen_agent.agents import GroupChat, GroupChatCreator
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui.gradio import gr, mgr
from qwen_agent.llm.schema import ContentItem, Message


def init_agent_service(cfgs):
    llm_cfg = {"model": "qwen2-72b-instruct", "generate_cfg": {"max_input_tokens": 31000}}
    bot = GroupChat(agents=cfgs, llm=llm_cfg)
    return bot


def init_agent_service_create():
    llm_cfg = {"model": "qwen2-72b-instruct", "generate_cfg": {"max_input_tokens": 31000}}
    bot = GroupChatCreator(llm=llm_cfg)
    return bot


# =========================================================
# Below is the gradio service: front-end and back-end logic
# =========================================================

app_global_para = {
    'messages': [],
    'messages_create': [],
    'is_first_upload': False,
    'uploaded_file': '',
    'user_interrupt': True
}

# Initialized group chat configuration
CFGS = {
    'background': 'openhis 云诊所开发群。',
    'agents': [
        {
            'name': '胡路',
            'description': '你是需求工程师，负责将客户的功能需传达给研发团队并回答团队关于需求的问题。（这是一个真实用户）',
            'is_human': True  # mark this as a real person
        },
        {
            'name': '周辉',
            'description': '前端工程师',
            'instructions': '你负责项目前端页面的开发，你使用的前端框架是 Vue.js，你会根据功能需求给出详细的前端代码实现。',
            'knowledge_files': [
                r'D:\workspace\mine\medical2.0\openhis-ui\src'
            ],
            'selected_tools': []
        },
        {
            'name': '坤超',
            'description': '后端工程师',
            'instructions': '你是负责项目后端API接口的开发，你使用的后端框架是 java+spring，你会根据功能需求给出详细的后端代码实现。',
            'knowledge_files': [
                r'D:\workspace\mine\medical2.0\openhis-api\src'
            ],
            'selected_tools': []
        },
        {
            'name': '大头',
            'description': '产品经理',
            'instructions': '你是资深产品经理，有丰富的医院信息化系统产品经验，熟悉openhis云诊所的功能，并能把需求转化为产品功能需求。',
            'knowledge_files': [
                r'D:\workspace\mine\medical2.0\云诊所管理系统操作手册.docx',
                r'D:\workspace\mine\medical2.0\云诊所管理系统说明书.pdf'
            ],
            'selected_tools': ['web_search']
        },
        {
            'name': '李难多',
            'description': '代码管理员',
            'instructions': '你负责对前后端提供的代码进行总结，指定代码的存储路径，并将代码写入到代码仓库里。',
            'knowledge_files': [
                r'D:\workspace\mine\medical2.0\openhis-ui\src',
                r'D:\workspace\mine\medical2.0\openhis-api\src'
            ],
            'selected_tools': []
        }
    ]
}

MAX_ROUND = 50


def app_tui():
    # Define a group chat agent from the CFGS
    bot = init_agent_service(CFGS)
    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append(Message('user', query, name="胡路"))
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)


def app(cfgs):
    # Todo: Reinstance every time or instance one time as global variable?
    cfgs = json5.loads(cfgs)
    bot = init_agent_service(cfgs=cfgs)

    # Record all mentioned agents: reply in order
    mentioned_agents_name = []

    for i in range(MAX_ROUND):
        messages = app_global_para['messages']
        print(i, messages)

        # Interrupt: there is new input from user
        if i == 0:
            app_global_para['user_interrupt'] = False
        if i > 0 and app_global_para['user_interrupt']:
            app_global_para['user_interrupt'] = False
            print('GroupChat is interrupted by user input!')
            # Due to the concurrency issue with Gradio, unable to call the second service simultaneously
            for rsp in app(json.dumps(cfgs, ensure_ascii=False)):
                yield rsp
            break
        # Record mentions into mentioned_agents_name list
        content = ''
        if messages:
            if isinstance(messages[-1].content, list):
                content = '\n'.join([x.text if x.text else '' for x in messages[-1].content]).strip()
            else:
                content = messages[-1].content.strip()
        if '@' in content:
            for x in content.split('@'):
                for agent in cfgs['agents']:
                    if x.startswith(agent['name']):
                        if agent['name'] not in mentioned_agents_name:
                            mentioned_agents_name.append(agent['name'])
                        break
        # Get one response from groupchat
        response = []
        try:
            display_history = _get_display_history_from_message()
            yield display_history
            for response in bot.run(messages, need_batch_response=False, mentioned_agents_name=mentioned_agents_name):
                if response:
                    if response[-1].content == PENDING_USER_INPUT:
                        # Stop printing the special message for mention human
                        break
                    incremental_history = []
                    for x in response:
                        function_display = ''
                        if x.function_call:
                            function_display = f'\nCall Function: {str(x.function_call)}'
                        incremental_history += [[None, f'{x.name}: {x.content}{function_display}']]
                    display_history = _get_display_history_from_message()
                    yield display_history + incremental_history

        except Exception as ex:
            raise ValueError(ex)

        if not response:
            # The topic ends
            print('No one wants to talk anymore!')
            break
        if mentioned_agents_name:
            assert response[-1].name == mentioned_agents_name[0]
            mentioned_agents_name.pop(0)

        if response and response[-1].content == PENDING_USER_INPUT:
            # Terminate group chat and wait for user input
            print('Waiting for user input!')
            break

        # Record the response to messages
        app_global_para['messages'].extend(response)


def app_test():
    app(cfgs=CFGS)


def app_create(history, now_cfgs):
    now_cfgs = json5.loads(now_cfgs)
    if not history:
        yield history, json.dumps(now_cfgs, indent=4, ensure_ascii=False)
    else:

        if len(history) == 1:
            new_cfgs = {'background': '', 'agents': []}
            # The first time to create grouchat
            exist_cfgs = now_cfgs['agents']
            for cfg in exist_cfgs:
                if 'is_human' in cfg and cfg['is_human']:
                    new_cfgs['agents'].append(cfg)
        else:
            new_cfgs = now_cfgs
        app_global_para['messages_create'].append(Message('user', history[-1][0].text))
        response = []
        try:
            agent = init_agent_service_create()
            for response in agent.run(messages=app_global_para['messages_create']):
                display_content = ''
                for rsp in response:
                    if rsp.name == 'role_config':
                        cfg = json5.loads(rsp.content)
                        old_pos = -1
                        for i, x in enumerate(new_cfgs['agents']):
                            if x['name'] == cfg['name']:
                                old_pos = i
                                break
                        if old_pos > -1:
                            new_cfgs['agents'][old_pos] = cfg
                        else:
                            new_cfgs['agents'].append(cfg)

                        display_content += f'\n\n{cfg["name"]}: {cfg["description"]}\n{cfg["instructions"]}'
                    elif rsp.name == 'background':
                        new_cfgs['background'] = rsp.content
                        display_content += f'\n群聊背景：{rsp.content}'
                    else:
                        display_content += f'\n{rsp.content}'

                history[-1][1] = display_content.strip()
                yield history, json.dumps(new_cfgs, indent=4, ensure_ascii=False)
        except Exception as ex:
            raise ValueError(ex)

        app_global_para['messages_create'].extend(response)


def _get_display_history_from_message():
    # Get display history from messages
    display_history = []
    for msg in app_global_para['messages']:
        if isinstance(msg.content, list):
            content = '\n'.join([x.text if x.text else '' for x in msg.content]).strip()
        else:
            content = msg.content.strip()
        function_display = ''
        if msg.function_call:
            function_display = f'\nCall Function: {str(msg.function_call)}'
        content = f'{msg.name}: {content}{function_display}'
        display_history.append((content, None) if msg.name == 'user' else (None, content))
    return display_history


def get_name_of_current_user(cfgs):
    for agent in cfgs['agents']:
        if 'is_human' in agent and agent['is_human']:
            return agent['name']
    return 'user'


def add_text(text, cfgs):
    app_global_para['user_interrupt'] = True
    content = [ContentItem(text=text)]
    if app_global_para['uploaded_file'] and app_global_para['is_first_upload']:
        app_global_para['is_first_upload'] = False  # only send file when first upload
        content.append(ContentItem(file=app_global_para['uploaded_file']))
    app_global_para['messages'].append(
        Message('user', content=content, name=get_name_of_current_user(json5.loads(cfgs))))

    return _get_display_history_from_message(), None


def chat_clear():
    app_global_para['messages'] = []
    return None


def chat_clear_create():
    app_global_para['messages_create'] = []
    return None, None


def add_file(file):
    app_global_para['uploaded_file'] = file.name
    app_global_para['is_first_upload'] = True
    return file.name


def add_text_create(history, text):
    history = history + [(text, None)]
    return history, gr.update(value='', interactive=False)


with gr.Blocks(theme='soft') as demo:
    display_config = gr.Textbox(
        label=  # noqa
        'Current GroupChat: (If editing, please maintain this JSON format)',
        value=json.dumps(CFGS, indent=4, ensure_ascii=False),
        interactive=True)

    with gr.Tab('Chat', elem_id='chat-tab'):
        with gr.Column():
            chatbot = mgr.Chatbot(elem_id='chatbot', height=750, show_copy_button=True, flushing=False)
            with gr.Row():
                with gr.Column(scale=3, min_width=0):
                    auto_speak_button = gr.Button('Randomly select an agent to speak first')
                    auto_speak_button.click(app, display_config, chatbot)
                with gr.Column(scale=10):
                    chat_txt = gr.Textbox(
                        show_label=False,
                        placeholder='Chat with Qwen...',
                        container=False,
                    )
                with gr.Column(scale=1, min_width=0):
                    chat_clr_bt = gr.Button('Clear')

            chat_txt.submit(add_text, [chat_txt, display_config], [chatbot, chat_txt],
                            queue=False).then(app, display_config, chatbot)

            chat_clr_bt.click(chat_clear, None, [chatbot], queue=False)

        demo.load(chat_clear, None, [chatbot], queue=False)

    with gr.Tab('Create', elem_id='chat-tab'):
        with gr.Column(scale=9, min_width=0):
            chatbot = mgr.Chatbot(elem_id='chatbot0', height=750, show_copy_button=True, flushing=False)
            with gr.Row():
                with gr.Column(scale=13):
                    chat_txt = gr.Textbox(
                        show_label=False,
                        placeholder='Chat with Qwen...',
                        container=False,
                    )
                with gr.Column(scale=1, min_width=0):
                    chat_clr_bt = gr.Button('Clear')

            txt_msg = chat_txt.submit(add_text_create, [chatbot, chat_txt], [chatbot, chat_txt],
                                      queue=False).then(app_create, [chatbot, display_config],
                                                        [chatbot, display_config])
            txt_msg.then(lambda: gr.update(interactive=True), None, [chat_txt], queue=False)

            chat_clr_bt.click(chat_clear_create, None, [chatbot, chat_txt], queue=False)
        demo.load(chat_clear_create, None, [chatbot, chat_txt], queue=False)

if __name__ == '__main__':
    # app_tui()
    demo.queue().launch()
