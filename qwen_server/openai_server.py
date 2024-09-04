import argparse
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import fastapi
import tiktoken
import uvicorn
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastchat.protocol.openai_api_protocol import ChatCompletionRequest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from qwen_agent.agents import Assistant, ReActChat
from qwen_agent.llm.schema import Message
from qwen_agent.tools.apibank import ApiBank


@asynccontextmanager
async def lifespan_context(app):
    yield


app = fastapi.FastAPI(lifespan=lifespan_context)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'],
                   allow_headers=['*'])

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").disabled = True
app.logger = logger = logging.getLogger(__name__)
app.startup_time = int(time.time())
app.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

apibank = ApiBank()


async def check_api_key(
        auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
) -> str:
    if auth is None or (token := auth.credentials) != app.args.served_api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
    return token


@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    return {
        "object": "list",
        "data": [
            {
                "id": app.args.served_model_name,
                "object": "model",
                "created": app.startup_time,
                "owned_by": "octopus_llm",
                "root": app.args.openai_model_name,
                "parent": None,
                "max_model_len": app.args.max_model_len,
                "permission": []
            }
        ]
    }


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    session_id = uuid.uuid4().hex
    messages = [Message(**msg) for msg in request.messages]

    previous_contents_cache = {}

    def generate_chunk_content(index, content):
        previous_content = previous_contents_cache.get(index, "")
        previous_contents_cache[index] = content

        return content[len(previous_content):]

    def response_generator():
        # agent = Assistant(llm={'model': 'qwen2-72b-instruct'})
        agent = ReActChat(llm={
                'model': 'qwen2-72b-instruct',
                'generate_cfg': {"max_input_tokens": 31000}
            },
            name='code interpreter',
            description='This agent can run code to solve the problem',
            function_list=apibank.functions)
        responses = agent.run(messages)

        for items in responses:
            json_data = ChatCompletionChunk(
                id=session_id,
                created=int(time.time()),
                model=app.args.served_model_name,
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        finish_reason=None,
                        delta=ChoiceDelta(role=item.role, content=generate_chunk_content(index, item.content))
                    )
                    for index, item in enumerate(items)
                ])
            yield "data: " + json_data.to_json(indent=None) + "\n\n"

        json_data = ChatCompletionChunk(
            id=session_id,
            created=int(time.time()),
            model=app.args.served_model_name,
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    delta=ChoiceDelta(role="assistant", content=None)
                )
            ])
        yield "data: " + json_data.to_json(indent=None) + "\n\n"

    return StreamingResponse(response_generator(), media_type='text/event-stream')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--openai-api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai-model-name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--served-api-key", type=str, default="sk-openai")
    parser.add_argument("--served-model-name", type=str, default="octopus-chat")
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--max-react-loops", type=int, default=20)
    parser.add_argument("--milvus-base-url", type=str, default="http://milvus.portal.clinify.cn")
    parser.add_argument("--tiktoken-encoding-name", type=str, default="cl100k_base")
    app.args = parser.parse_args()

    os.environ["DASHSCOPE_API_KEY"] = app.args.openai_api_key
    if app.tiktoken_encoder.name != app.args.tiktoken_encoding_name:
        app.tiktoken_encoder = tiktoken.get_encoding(app.args.tiktoken_encoding_name)

    uvicorn.run(app, host=app.args.host, port=app.args.port)
