from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import httpx
import json
import time
import config
from database import init_db, validate_api_key, log_usage


################### pydantic model #####################
class Message(BaseModel):
    content: str = Field(min_length=1)
    role: Literal["system", "user", "assistant"]


# ChatRequest is a list of messages{role and content}
class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None  # optional
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False


################ fastapi app ##################
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # one shared client for every request (connection pooling)
    app.state.http_client = httpx.AsyncClient(
        base_url=config.LLAMA_SERVER_URL, timeout=config.REQUEST_TIMEOUT
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="aihost", lifespan=lifespan)
bearer_scheme = HTTPBearer()


################ auth ##################
# returns (id, name) of the api key, or 401
def require_api_key(
    token: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
):
    key_record = validate_api_key(token.credentials)
    if key_record is None:
        raise HTTPException(status_code=401, detail="invalid api key")
    return key_record


def get_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def upstream_error(exc: Exception) -> HTTPException:
    if isinstance(exc, httpx.ConnectError):  # if the server is off
        return HTTPException(status_code=503, detail="Model Server is unavailable")
    if isinstance(exc, httpx.TimeoutException):
        return HTTPException(
            status_code=504, detail="Model Server took too long to respond"
        )
    return HTTPException(status_code=502, detail="Model server returned an error")


def log_request(key_record, status_code, usage, start_time):
    usage = usage or {}
    log_usage(
        api_key_id=key_record[0],
        status_code=status_code,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        latency_ms=(time.perf_counter() - start_time) * 1000,
    )


@app.get("/health")
async def health(client: Annotated[httpx.AsyncClient, Depends(get_client)]):
    # also tell the caller whether the model server is reachable
    try:
        response = await client.get("/health", timeout=5.0)
        model_server = "okay" if response.status_code == 200 else "loading"
    except httpx.HTTPError:
        model_server = "unavailable"
    return {"status": "okay", "model_server": model_server}


@app.get("/v1/models")
async def list_models(
    key_record: Annotated[tuple, Depends(require_api_key)],
    client: Annotated[httpx.AsyncClient, Depends(get_client)],
):
    try:
        response = await client.get("/v1/models")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise upstream_error(exc)
    return response.json()


########## request to local llm to return data ##########
@app.post("/v1/chat/completions")
async def chat_completions(
    chatrequest: ChatRequest,
    key_record: Annotated[tuple, Depends(require_api_key)],
    client: Annotated[httpx.AsyncClient, Depends(get_client)],
):
    # HTTPX needs JSON instead of Pydantic model
    payload = chatrequest.model_dump(
        exclude_none=True
    )  # Exclude_none will ignore optional value
    start_time = time.perf_counter()

    if chatrequest.stream:
        # ask the server to send token usage in the last chunk so we can log it
        payload["stream_options"] = {"include_usage": True}
        return await stream_chat(payload, key_record, client, start_time)

    try:
        response = await client.post("/v1/chat/completions", json=payload)
    except httpx.HTTPError as exc:
        error = upstream_error(exc)
        log_request(key_record, error.status_code, None, start_time)
        raise error

    if response.status_code >= 500:
        log_request(key_record, 502, None, start_time)
        raise HTTPException(status_code=502, detail="Model server returned an error")
    if response.status_code >= 400:
        # bad request (e.g. prompt too long) -> pass the model server's error on
        log_request(key_record, response.status_code, None, start_time)
        return JSONResponse(status_code=response.status_code, content=response.json())

    data = response.json()
    log_request(key_record, response.status_code, data.get("usage"), start_time)
    return data


async def stream_chat(payload, key_record, client, start_time):
    request = client.build_request("POST", "/v1/chat/completions", json=payload)
    try:
        response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        error = upstream_error(exc)
        log_request(key_record, error.status_code, None, start_time)
        raise error

    if response.status_code >= 400:
        body = await response.aread()
        await response.aclose()
        log_request(key_record, response.status_code, None, start_time)
        if response.status_code >= 500:
            raise HTTPException(
                status_code=502, detail="Model server returned an error"
            )
        return JSONResponse(status_code=response.status_code, content=json.loads(body))

    async def event_stream():
        usage = None
        status_code = 200
        try:
            # forward server-sent events line by line, peeking for usage
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[len("data: ") :])
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                    except json.JSONDecodeError:
                        pass
                yield line + "\n"
        except httpx.HTTPError:
            status_code = 502
        finally:
            await response.aclose()
            log_request(key_record, status_code, usage, start_time)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
