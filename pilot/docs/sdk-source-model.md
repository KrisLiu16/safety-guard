# aster_flow.model

连接 Run 中配置的模型，检查连通性，或直接发送模型请求。

## 源码

```python
"""连接 Run 中配置的模型，检查连通性，或直接发送模型请求。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from aiohttp import ClientSession, ClientTimeout, UnixConnector, web
from aster_runtime.sidecar import (
    AGENT_TOKEN,
    GATEWAY_LOG_FILE,
    GATEWAY_MAX_BODY_BYTES,
    GATEWAY_MAX_RESPONSE_BYTES,
    STREAM_CHUNK_BYTES,
    Upstream,
    model_path,
    select_protocol,
)
from aster_sidecar.gateway.models import CaptureFailed, ModelGateways

from . import model_check
from .errors import ModelRejected, TempFail
from .model_check import ModelCheck
from .retry import retry

if TYPE_CHECKING:
    from .runtime import Runtime
    from .sandbox import Sandbox
    from .sidecar import Sidecars


@dataclass(frozen=True, slots=True)
class Endpoint:
    """供指定沙箱使用的模型地址、凭证、协议和请求头。"""

    base_url: str
    api_key: str
    wire_api: str
    headers: dict[str, str]


class Model:
    """Run 中的一份模型配置，通过 ctx.models[名称] 获取。请求体使用该模型的原生协议。"""

    def __init__(self, runtime: Runtime, key: str, sidecars: Sidecars) -> None:
        self._runtime, self._sidecars, self.key = runtime, sidecars, key
        frozen = runtime.start.models[key]
        self.name, self.wire_api = frozen.name, frozen.wire_api
        self.access = runtime.start.model_access
        self.base_url = frozen.base_url if self.access == "direct" else ""
        self.api_key = frozen.api_key if self.access == "direct" else AGENT_TOKEN
        self.headers = dict(frozen.headers) if self.access == "direct" else {}
        self.tito_enabled = key in runtime.start.tito.models

    def connect(
        self,
        accepts: list[str] | tuple[str, ...],
        *,
        sandbox: Sandbox,
        session_id: str | None = None,
    ) -> Endpoint:
        """返回沙箱可用的模型连接信息。

        ``accepts`` 列出脚手架支持的协议。优先使用模型配置中的协议；开启 TITO 时，也可选择
        TITO 支持的其他协议。没有共同协议时抛 ValueError。
        gateway 模式下会确保沙箱中的 sidecar 已启动。预检发现后端或凭证不符合要求时抛
        ModelRejected；连接故障或服务端错误抛 TempFail。
        """
        wire = select_protocol(
            list(accepts), self.wire_api, tito=self.tito_enabled and self.access == "gateway"
        )
        if self.access == "direct":
            return Endpoint(self.base_url, self.api_key, wire, dict(self.headers))
        if sandbox.spec.model is None:
            raise ModelRejected("该环境没有启用模型访问，请显式配置模型能力")
        handle = self._sidecars.ensure(sandbox)
        identity = session_id or uuid.uuid4().hex
        route = f"/models/{quote(self.key, safe='')}/sessions/{quote(identity, safe='')}"
        return Endpoint(handle.base_url + route, AGENT_TOKEN, wire, {})

    # safeguard: SG-14
    def check(
        self,
        timeout: float = 60.0,
        *,
        sandbox: Sandbox | None = None,
        retry_seconds: float = 600.0,
    ) -> ModelCheck:
        """检查模型是否可调用；传入 sandbox 时，通过该沙箱的模型入口检查。

        一次探测最多等 ``timeout`` 秒。超时、连不上、429、5xx 按 Worker 给的间隔退避重探（SG-14），从第一次探测算起
        ``retry_seconds`` 秒内（最后一次探测也在这之内结束）还不通就抛 ``TempFail``；``retry_seconds`` 不大于 ``timeout``
        时只探一次。其余 4xx 直接抛 ``ModelRejected``，不重探。在 Agent 启动之前调用时，重探的时间算在 ``setup_timeout`` 里。
        """
        tries = 0
        started = time.monotonic()

        def probe() -> ModelCheck:
            nonlocal tries
            tries += 1
            return self._probe(timeout, sandbox)

        try:
            return retry(
                self._runtime,
                probe,
                when=lambda exc: isinstance(exc, TempFail),
                what=f"model check of {self.key}",
                until=started + retry_seconds - timeout,
            )
        except TempFail as exc:
            if tries == 1:
                raise
            elapsed = time.monotonic() - started
            raise TempFail(f"{exc}；{tries} 次探测、{elapsed:.0f} 秒内都没通过") from None

    def _probe(self, timeout: float, sandbox: Sandbox | None) -> ModelCheck:
        if sandbox is None or self.access == "direct":
            frozen = self._runtime.start.models[self.key]
            return model_check.check(
                frozen.base_url,
                frozen.api_key,
                self.name,
                self.wire_api,
                dict(frozen.headers),
                timeout,
            )
        result = self._sidecars.control(
            sandbox,
            {"op": "check", "model": self.key, "wire_api": self.wire_api, "timeout": timeout},
        )
        status, seconds = int(result["status"]), float(result["seconds"])
        if status == 0:
            raise TempFail(f"模型自检 {seconds:g} 秒内没有收到响应（超时或连不上）")
        if status == 429 or status >= 500:
            raise TempFail(f"模型自检失败，状态 {status}")
        if status >= 400:
            raise ModelRejected(f"模型自检被拒绝，状态 {status}")
        return ModelCheck(self.connect([self.wire_api], sandbox=sandbox).base_url, status, seconds)

    async def _chunks(
        self,
        body: dict[str, Any],
        timeout: float,  # noqa: ASYNC109 - timeout controls the upstream transport
        *,
        stream: bool,
    ) -> AsyncGenerator[bytes]:
        limit = min(timeout, self._runtime.remaining_seconds())
        if limit <= 0:
            raise TimeoutError("流程模型调用已超过总时限")
        execution_id = uuid.uuid4().hex
        state = Path(self._runtime.start.tmp) / f"model-{execution_id}"
        state.mkdir()
        gateway = ModelGateways(
            {self.key: Upstream(**self._runtime.start.models[self.key].model_dump())},
            state,
            tito_models={self.key} if self.tito_enabled else set(),
            # 流程进程里的调用不经沙箱，这个状态目录不收回来，调用原文只由沙箱里的网关记。
            record_calls=False,
            model_enabled=True,
        )
        app = web.Application(client_max_size=GATEWAY_MAX_BODY_BYTES)
        app.router.add_route("*", "/{path:.*}", gateway.handle)
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=1)
        socket_dir = TemporaryDirectory(prefix="aster-model-")
        socket_path = str(Path(socket_dir.name) / "gateway.sock")
        self._runtime.channel.send(
            "phase",
            phase="execution_started",
            execution_id=execution_id,
            model=self.key,
            execution_kind="model",
        )
        try:
            try:
                await gateway.start()
            except CaptureFailed as exc:
                raise (TempFail if exc.retryable else ModelRejected)(str(exc)) from exc
            await runner.setup()
            await web.UnixSite(runner, socket_path).start()
            async with (
                ClientSession(
                    connector=UnixConnector(path=socket_path), timeout=ClientTimeout(total=limit)
                ) as client,
                client.post(
                    f"http://model/models/{quote(self.key, safe='')}/sessions/{execution_id}"
                    + model_path(self.wire_api, self.name, stream=stream),
                    json=body,
                    headers=(
                        {"anthropic-version": "2023-06-01"}
                        if self.wire_api == "anthropic_messages"
                        else {}
                    ),
                ) as response,
            ):
                if response.status >= 400:
                    raise ModelRejected(f"模型请求失败，HTTP {response.status}")
                async for chunk in response.content.iter_chunked(STREAM_CHUNK_BYTES):
                    yield chunk
        finally:
            await gateway.close()
            await runner.cleanup()
            socket_dir.cleanup()
            # 这次调用独占这个状态目录，请求日志整份就是这一次的。
            requests = await asyncio.to_thread(_read_if_present, state / GATEWAY_LOG_FILE)
            if requests:
                self._runtime.log("sidecar", requests)
            self._runtime.channel.send("artifact", sandbox_id=None, path=str(state))
            self._runtime.channel.send(
                "phase",
                phase="execution_finished",
                execution_id=execution_id,
                model=self.key,
                execution_kind="model",
                requests=gateway.health()["requests"],
            )

    async def _request(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:  # noqa: ASYNC109 - upstream transport limit
        data = bytearray()
        async for chunk in self._chunks(body, timeout, stream=False):
            data.extend(chunk)
            if len(data) > GATEWAY_MAX_RESPONSE_BYTES:
                raise ValueError(f"模型响应超过 {GATEWAY_MAX_RESPONSE_BYTES // 1024**2} MiB")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("模型没有返回 JSON 对象")
        return result

    def request(self, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """直接请求模型；body 是该协议的原生 JSON，流式响应使用 stream。Gemini 打 ``:generateContent``。"""
        return self._runtime.loop.call(self._request(body, timeout))

    async def arequest(self, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:  # noqa: ASYNC109 - upstream transport limit
        """异步原生请求，取消关闭该请求的连接与网关。"""
        return await self._request(body, timeout)

    def stream(self, body: dict[str, Any], *, timeout: float) -> Iterator[bytes]:
        """返回响应字节流；OpenAI 和 Anthropic 由调用者在请求里显式设置 stream 字段，Gemini 打 ``:streamGenerateContent?alt=sse``。"""
        source = self._chunks(body, timeout, stream=True)
        try:
            while True:
                try:
                    yield self._runtime.loop.call(_next(source))
                except StopAsyncIteration:
                    return
        finally:
            self._runtime.loop.call(_close(source))

    async def astream(self, body: dict[str, Any], *, timeout: float) -> AsyncIterator[bytes]:  # noqa: ASYNC109 - upstream transport limit
        """异步字节流，关闭迭代时回收这次请求。"""
        source = self._chunks(body, timeout, stream=True)
        try:
            async for chunk in source:
                yield chunk
        finally:
            await source.aclose()


async def _next(source: AsyncGenerator[bytes]) -> bytes:
    return await anext(source)


def _read_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


async def _close(source: AsyncGenerator[bytes]) -> None:
    await source.aclose()
```
