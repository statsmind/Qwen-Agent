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
    'user_interrupt': False
}

# Initialized group chat configuration
CFGS = {
    'background': 'openhis 云诊所开发项目组，需求工程师给出需求后，项目组其他工程师完成相关的代码。',
    'agents': [
        {
            'name': '舒高伟',
            'description': '一位需求工程师(这是一个真实用户)',
            'instructions': '负责把客户的需求传达给项目组进行开发。',
            'is_human': True,
            'should_interrupt': False
        },
        {
            'name': '张庭玮',
            'description': '一位前端工程师',
            'instructions': '项目目前使用的前端框架是 Vue.js，你的任务是根据功能需求和已有代码开发前端页面代码。',
            'llm': {
                'model': 'codeqwen1.5-7b-chat',
                'generate_cfg': {
                    'max_input_tokens': 31000
                }
            },
            'knowledge_files': [
                {
                    "path": r'D:\workspace\mine\medical2.0\openhis-ui\src',
                    "excludes": [
                        r'\/resources\/',
                        r'\/lang\/',
                        r'resource\.js'
                    ]
                }
            ],
            'selected_tools': []
        },
        {
            'name': '李明明',
            'description': '一位后端工程师',
            'instructions': '项目目前使用的后端框架是 spring boot，你的任务是根据功能需求和已有代码开发后端服务及接口代码。',
            'llm': {
                'model': 'codeqwen1.5-7b-chat',
                'generate_cfg': {
                    'max_input_tokens': 31000
                }
            },
            'knowledge_files': [
                {
                    "path": r'D:\workspace\mine\medical2.0\openhis-api\src',
                    "excludes": [
                        r'\/resources\/',
                        r'resource\.js'
                    ]
                }
            ],
            'selected_tools': []
        },
        {
            'name': '安歌',
            'description': '一位数据库DBA',
            'instructions': '负责根据功能需求进行数据库表结构的创建及修改。',
            'llm': {
                'model': 'codeqwen1.5-7b-chat',
                'generate_cfg': {
                    'max_input_tokens': 31000
                }
            },
            'knowledge_files': [
                r'D:\workspace\mine\medical2.0\schema.sql'
            ],
            'selected_tools': []
        },
        # {
        #     'name': '刘龙刚',
        #     'description': '一位测试工程师',
        #     'instructions': '任务是根据功能需求进行测试用例开发，你使用的工具是 Intellij Idea Http Client。',
        #     'knowledge_files': [
        #         'https://blog.csdn.net/millery22/article/details/123566322'
        #     ],
        #     'selected_tools': []
        # },
        # {
        #     'name': '王鑫',
        #     'description': '一位项目经理',
        #     'instructions': '有丰富的医院信息化系统项目管理经验，熟悉openhis云诊所的功能，你安排团队成员的工作。你不需要向任何人询问关于需求的信息，你决定一切。',
        #     'knowledge_files': [
        #         r'D:\workspace\mine\medical2.0\云诊所管理系统操作手册.docx',
        #         r'D:\workspace\mine\medical2.0\云诊所管理系统说明书.pdf'
        #     ],
        #     'selected_tools': []
        # }
    ]
}

MAX_ROUND = 50


def app_tui():
    # Define a group chat agent from the CFGS
    bot = init_agent_service(CFGS)
    message_queue = [
        '客户想在患者登记页去掉身份证录入'
    ]
    # Chat
    messages = []
    while True:
        if len(message_queue) > 0:
            query = message_queue.pop()
        else:
            query = input('user question: ')

        messages.append(Message('user', query, '舒高伟'))
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
            parts = content.split('@')
            if len(parts) > 1:
                parts = parts[1:]

                for x in parts:
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
