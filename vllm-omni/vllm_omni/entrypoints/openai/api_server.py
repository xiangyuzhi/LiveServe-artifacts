# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import base64
import copy
import dataclasses
import io
import json
import multiprocessing
import multiprocessing.forkserver as forkserver
import os

# Image generation API imports
import random
import time
from argparse import Namespace
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from numbers import Integral
from typing import Annotated, Any, Literal, cast

import httpx
import numpy as np
import vllm.entrypoints.launcher as vllm_launcher
import vllm.envs as envs
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi import routing as fastapi_routing
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.datastructures import State
from starlette.routing import Mount, Route, WebSocketRoute
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http, terminate_if_errored
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.mcp.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.openai.api_server import build_app as build_openai_app
from vllm.entrypoints.openai.api_server import setup_server as setup_openai_server
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.realtime.serving import OpenAIServingRealtime
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.server_utils import get_uvicorn_log_config
from vllm.entrypoints.openai.speech_to_text.serving import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding as OpenAIServingEmbedding
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.serve.disagg.serving import ServingTokens

# vLLM moved `base` from openai.basic.api_router to serve.instrumentator.basic.
# Keep a fallback for older/newer upstream layouts during rebase windows.
from vllm.entrypoints.serve.instrumentator.basic import base
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import (
    create_error_response,
    load_aware_call,
    process_lora_modules,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.tasks import POOLING_TASKS
from vllm.tool_parsers import ToolParserManager
from vllm.utils import random_uuid
from vllm.utils.system_utils import decorate_logs
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.image_api_utils import (
    SUPPORTED_LAYERED_RESOLUTIONS,
    encode_image_base64,
    parse_size,
    validate_layered_layers,
)
from vllm_omni.entrypoints.openai.protocol.audio import (
    BatchSpeechRequest,
    OpenAICreateAudioGenerateRequest,
    OpenAICreateSpeechRequest,
)
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from vllm_omni.entrypoints.openai.protocol.videos import (
    SecondStr,
    SizeStr,
    VideoDeleteResponse,
    VideoError,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoListResponse,
    VideoResponse,
)
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.openai.serving_audio_generate import OmniOpenAIServingAudioGenerate
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo, ReferenceImage
from vllm_omni.entrypoints.openai.serving_video_stream import OmniStreamingVideoHandler
from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    get_default_sampling_params_list,
)
from vllm_omni.entrypoints.openai.storage import STORAGE_MANAGER
from vllm_omni.entrypoints.openai.stores import VIDEO_STORE, VIDEO_TASKS
from vllm_omni.entrypoints.openai.utils import get_stage_type, parse_lora_request
from vllm_omni.entrypoints.openai.video_api_utils import decode_input_reference
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.profiler.interaction_trace import emit
from vllm_omni.profiler.interaction_trace import enabled as trace_enabled

logger = init_logger(__name__)
router = APIRouter()

MAX_UINT32_SEED = 2**32 - 1
profiler_router = APIRouter()


def _should_enable_profiler_endpoints(stage_configs: list | None) -> bool:
    """Check if any stage has profiler_config set in its engine_args."""
    if not stage_configs:
        return False
    for stage in stage_configs:
        engine_args = stage.get("engine_args") if isinstance(stage, dict) else getattr(stage, "engine_args", None)
        if engine_args is None:
            continue
        profiler_config = (
            engine_args.get("profiler_config")
            if isinstance(engine_args, dict)
            else getattr(engine_args, "profiler_config", None)
        )
        if profiler_config is not None:
            profiler = (
                profiler_config.get("profiler")
                if isinstance(profiler_config, dict)
                else getattr(profiler_config, "profiler", None)
            )
            if profiler is not None:
                return True
    return False


class ProfileRequest(BaseModel):
    """Request model for profiling endpoints."""

    stages: list[int] | None = Field(
        default=None,
        description="List of stage IDs to profile. If None, profiles all stages.",
    )


def _remove_route_from_router(
    router: APIRouter,
    path: str,
    methods: set[str] | None = None,
) -> None:
    methods_set = {method.upper() for method in methods} if methods else None
    for route in list(router.routes):
        if getattr(route, "path", None) != path:
            continue
        if methods_set is not None:
            route_methods = {method.upper() for method in (getattr(route, "methods", None) or set())}
            if not (route_methods & methods_set):
                continue
        router.routes.remove(route)


ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


async def _get_vllm_config(engine_client: EngineClient) -> Any:
    if hasattr(engine_client, "get_vllm_config"):
        return await engine_client.get_vllm_config()
    return getattr(engine_client, "vllm_config", None)


def _engine_state_for_log(engine: Any) -> str:
    parts: list[str] = []
    for name in ("errored", "is_running"):
        try:
            value = getattr(engine, name)
        except Exception as exc:
            value = f"<{type(exc).__name__}: {exc}>"
        parts.append(f"{name}={value}")

    inner = getattr(engine, "engine", None)
    if inner is not None:
        try:
            parts.append(f"orchestrator_alive={inner.is_alive()}")
        except Exception as exc:
            parts.append(f"orchestrator_alive=<{type(exc).__name__}: {exc}>")

    final_task = getattr(engine, "final_output_task", None)
    if final_task is not None:
        try:
            parts.append(f"final_output_done={final_task.done()}")
            if final_task.done():
                parts.append(f"final_output_cancelled={final_task.cancelled()}")
                exc = final_task.exception() if not final_task.cancelled() else None
                if exc is not None:
                    parts.append(f"final_output_exc={type(exc).__name__}: {exc}")
        except Exception as exc:
            parts.append(f"final_output_state=<{type(exc).__name__}: {exc}>")

    stage_bits: list[str] = []
    for idx, stage_client in enumerate(getattr(engine, "stage_clients", []) or []):
        resources = getattr(stage_client, "resources", None)
        proc = getattr(stage_client, "_proc", None)
        try:
            proc_alive = proc.is_alive() if proc is not None else None
            proc_exit = proc.exitcode if proc is not None else None
        except Exception:
            proc_alive = proc_exit = "?"
        stage_bits.append(
            "stage%s(engine_dead=%s,proc_alive=%s,exit=%s)"
            % (
                idx,
                getattr(resources, "engine_dead", getattr(stage_client, "_engine_dead", None)),
                proc_alive,
                proc_exit,
            )
        )
    if stage_bits:
        parts.append("stages=[" + ", ".join(stage_bits) + "]")
    return " ".join(parts)


def _install_omni_watchdog_logging() -> None:
    original = vllm_launcher.terminate_if_errored
    if getattr(original, "_omni_watchdog_logging", False):
        return

    def _logged_terminate_if_errored(server: Any, engine: Any) -> None:
        try:
            if getattr(engine, "errored", False) and not getattr(engine, "is_running", False):
                logger.error("Omni watchdog is shutting down HTTP server: %s", _engine_state_for_log(engine))
        except Exception:
            logger.exception("Failed to inspect Omni engine state before watchdog shutdown")
        return original(server, engine)

    _logged_terminate_if_errored._omni_watchdog_logging = True  # type: ignore[attr-defined]
    vllm_launcher.terminate_if_errored = _logged_terminate_if_errored


def _remove_route_from_app(app, path: str, methods: set[str] | None = None):
    """Remove matching HTTP methods without mutating shared included routers.

    Older FastAPI versions flatten includes; newer versions retain router
    branches with their own prefixes and cached effective routes. Copy changed
    branches so another app or prefix using the same router remains intact.
    """
    included_type = getattr(fastapi_routing, "_IncludedRouter", None)

    def prune(owner, prefix, ancestors):
        if id(owner) in ancestors:
            raise RuntimeError("Cannot override a route in a cyclic router tree")
        ancestors = ancestors | {id(owner)}
        routes = []
        changed = False
        for route in owner.routes:
            if included_type is not None and isinstance(route, included_type):
                child = route.original_router
                context = route.include_context
                if (
                    not isinstance(child, APIRouter)
                    or not dataclasses.is_dataclass(context)
                    or not isinstance(context.prefix, str)
                    or context.included_router is not child
                    or not callable(getattr(child, "_mark_routes_changed", None))
                ):
                    raise RuntimeError("Unsupported FastAPI included router layout")
                replacement = prune(child, prefix + context.prefix, ancestors)
                if replacement is not child:
                    # A fresh wrapper is essential: copying one retains its
                    # effective-route cache and lock from the original branch.
                    route = included_type(
                        original_router=replacement,
                        include_context=dataclasses.replace(context, included_router=replacement),
                    )
                    changed = True
            elif isinstance(route, Route):
                if prefix + route.path == path and route.methods is None and methods is not None:
                    raise RuntimeError(f"Cannot partially override an all-method route: {path!r}")
                if prefix + route.path == path and (methods is None or route.methods & methods):
                    remaining = route.methods - methods if methods is not None else set()
                    changed = True
                    if not remaining:
                        continue
                    route = copy.copy(route)
                    route.methods = remaining
            elif isinstance(route, WebSocketRoute):
                pass  # HTTP overrides must not remove websocket endpoints.
            elif isinstance(route, Mount):
                mounted_path = prefix + route.path
                if "{" in mounted_path:
                    raise RuntimeError("Cannot override routes with parameterized ASGI mounts")
                if path == mounted_path or path.startswith(mounted_path + "/"):
                    raise RuntimeError(f"Cannot override {path!r} inside a mounted ASGI app")
            else:
                raise RuntimeError(f"Unsupported route type during override: {type(route).__name__}")
            routes.append(route)
        if not changed:
            return owner
        replacement = copy.copy(owner)
        replacement.routes = routes
        mark_changed = getattr(replacement, "_mark_routes_changed", None)
        if callable(mark_changed):
            mark_changed()
        return replacement

    replacement = prune(app.router, "", set())
    if replacement is not app.router:
        app.router.routes = replacement.routes
        mark_changed = getattr(app.router, "_mark_routes_changed", None)
        if callable(mark_changed):
            mark_changed()
        app.openapi_schema = None


def _register_omni_exception_handlers(app) -> None:
    """Override upstream vLLM exception handlers with Omni-aware versions.

    The upstream ``engine_error_handler`` is designed for ``AsyncLLM`` (single
    EngineCore process).  Omni uses a multi-stage orchestrator with different
    health semantics, so we register our own handlers that:

    - Log multi-stage diagnostic info (orchestrator liveness, per-stage health)
      when an ``EngineDeadError`` is caught.
    - Call ``terminate_if_errored``
    - Return an OpenAI-compatible error JSON response.
    """

    async def omni_engine_error_handler(
        req: Request,
        exc: EngineDeadError | EngineGenerateError,
    ):
        request_id = _get_request_id_from_request(req)

        if req.app.state.args.log_error_stack:
            logger.exception("Engine Exception caught. Request id: %s", request_id)

        return _create_engine_error_json_response(req, exc)

    app.exception_handler(EngineGenerateError)(omni_engine_error_handler)
    app.exception_handler(EngineDeadError)(omni_engine_error_handler)


def _get_request_id_from_request(req: Request) -> str | None:
    return req.state.request_metadata.request_id if hasattr(req.state, "request_metadata") else None


def _build_engine_error_payload(
    exc: EngineDeadError | EngineGenerateError,
    *,
    request_id: str | None,
) -> tuple[dict[str, Any], int]:
    err = create_error_response(exc)
    payload = err.model_dump()
    error_body = payload.get("error", {})

    error_body["request_id"] = request_id
    error_body["error_stage_id"] = getattr(exc, "error_stage_id", None)

    return payload, err.error.code


def _create_engine_error_json_response(
    req: Request,
    exc: EngineDeadError | EngineGenerateError,
) -> JSONResponse:
    request_id = _get_request_id_from_request(req)
    error_stage_id = getattr(exc, "error_stage_id", None)
    engine = req.app.state.engine_client

    if isinstance(exc, EngineDeadError):
        # Log Omni-specific diagnostic information for dead engines.
        orchestrator_alive = engine.engine.is_alive() if hasattr(engine, "engine") else "N/A"
        logger.error(
            "EngineDeadError: orchestrator_alive=%s, errored=%s, request_id=%s, error_stage_id=%s",
            orchestrator_alive,
            engine.errored,
            request_id,
            error_stage_id,
        )

    terminate_if_errored(
        server=req.app.state.server,
        engine=engine,
    )

    payload, status_code = _build_engine_error_payload(exc, request_id=request_id)
    return JSONResponse(content=payload, status_code=status_code)


class _DiffusionServingModels:
    """Minimal OpenAIServingModels implementation for diffusion-only servers.

    vLLM's /v1/models route expects `app.state.openai_serving_models` to expose
    `show_available_models()`. In pure diffusion mode we don't initialize the
    full OpenAIServingModels (it depends on LLM-specific processors), so we
    provide a lightweight fallback.
    """

    class _NullModelConfig:
        def __getattr__(self, name):
            return None

    class _Unsupported:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, *args, **kwargs):
            raise NotImplementedError(f"{self.name} is not supported in diffusion mode")

        def __getattr__(self, attr):
            raise NotImplementedError(f"{self.name}.{attr} is not supported in diffusion mode")

    def __init__(self, base_model_paths: list[BaseModelPath]) -> None:
        self._base_model_paths = base_model_paths
        self.model_config = self._NullModelConfig()

    @property
    def base_model_paths(self) -> list[BaseModelPath]:
        return self._base_model_paths

    def is_base_model(self, model_name: str) -> bool:
        return any(p.name == model_name for p in self._base_model_paths)

    def __getattr__(self, name):
        return self._Unsupported(name)

    async def show_available_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self._base_model_paths
            ]
        )


# Server entry points


async def omni_run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server.

    Unified entry point that automatically handles both LLM and Diffusion models
    through AsyncOmni, which manages multi-stage pipelines.
    """
    # Suppress Pydantic serialization warnings globally for multimodal content
    # (e.g., when ChatMessage.content is a list instead of str)
    import warnings as warnings_module

    warnings_module.filterwarnings("ignore", message=".*Pydantic.*serialization.*", category=UserWarning)
    warnings_module.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer")

    listen_address, sock = setup_openai_server(args)

    # Unified use of omni_run_server_worker, AsyncOmni automatically handles LLM and Diffusion models
    await omni_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def omni_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        from vllm.reasoning import ReasoningParserManager

        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    # Load logging config for uvicorn if specified
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_omni(
        args,
        client_config=client_config,
    ) as engine_client:
        supported_tasks: tuple[str, ...]
        if hasattr(engine_client, "get_supported_tasks"):
            supported_tasks = tuple(await engine_client.get_supported_tasks())
        else:
            supported_tasks = ("generate",)
        if not supported_tasks:
            # Only default to "generate" when get_supported_tasks is not implemented;
            # TTS-only models intentionally return an empty set.
            if not hasattr(engine_client, "get_supported_tasks"):
                supported_tasks = ("generate",)

        # OMNI: Pass supported_tasks to build_app (required by upstream vLLM)
        app = build_openai_app(args, supported_tasks)
        # OMNI: Remove upstream routes that we override with omni-specific handlers
        _remove_route_from_app(app, "/v1/chat/completions", {"POST"})
        _remove_route_from_app(app, "/v1/models", {"GET"})  # Remove upstream /v1/models to use omni's handler
        app.include_router(router)

        # OMNI: Override upstream exception handlers with Omni-aware versions
        # that understand the multi-stage orchestrator lifecycle.
        _register_omni_exception_handlers(app)

        await omni_init_app_state(engine_client, app.state, args)

        # Conditionally register profiler endpoints based on stage YAML configs
        stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        if _should_enable_profiler_endpoints(stage_configs):
            logger.warning("Profiler endpoints are enabled. This should ONLY be used for local development!")
            app.include_router(profiler_router)

        vllm_config = await _get_vllm_config(engine_client)

        # Check if pure diffusion mode (vllm_config will be None)
        is_pure_diffusion = vllm_config is None
        if is_pure_diffusion:
            logger.info(
                "Starting vLLM API server (pure diffusion mode) on %s",
                listen_address,
            )
        else:
            logger.info(
                "Starting vLLM API server %d on %s",
                vllm_config.parallel_config._api_process_rank,
                listen_address,
            )
        _install_omni_watchdog_logging()
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )
        logger.warning("serve_http returned; engine_state=%s", _engine_state_for_log(engine_client))

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        state = getattr(app, "state", None)
        serving_speech = getattr(state, "openai_serving_speech", None) if state is not None else None
        if serving_speech is not None:
            serving_speech.shutdown()
        sock.close()


@asynccontextmanager
async def build_async_omni(
    args: Namespace,
    *,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """Build an AsyncOmni instance from command-line arguments.

    Creates an async context manager that yields an AsyncOmni instance
    configured from the provided arguments. Handles forkserver setup if
    needed and ensures proper cleanup on exit.

    Args:
        args: Parsed command-line arguments containing model and configuration
        disable_frontend_multiprocessing: Optional flag to disable frontend
            multiprocessing
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use
    """
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle async_omni lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    async with build_async_omni_from_stage_config(
        args,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
    ) as async_omni:
        yield async_omni


@asynccontextmanager
async def build_async_omni_from_stage_config(
    args: Namespace,
    *,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """Create AsyncOmni from stage configuration.

    Creates an AsyncOmni instance either in-process or using multiprocess
    RPC. Loads stage configurations from the model or from a specified path.

    Args:
        args: Parsed command-line arguments containing model and stage configs
        disable_frontend_multiprocessing: Flag to disable frontend multiprocessing
            for compatibility with existing CLI options
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use

    Note:
        Stage configurations are loaded from args.stage_configs_path if provided,
        otherwise from the model's default configuration.
    """

    if disable_frontend_multiprocessing:
        logger.warning("Ignoring --disable-frontend-multiprocessing for AsyncOmni runtime.")

    async_omni: EngineClient | None = None

    try:
        kwargs = vars(args).copy()
        kwargs.pop("model", None)
        async_omni = AsyncOmni(model=args.model, **kwargs)

        # # Don't keep the dummy data in memory
        # await async_llm.reset_mm_cache()

        yield async_omni
    finally:
        if async_omni:
            async_omni.shutdown()


async def omni_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
) -> None:
    """Initialize the FastAPI application state for omni API server.

    Sets up the application state with model information, request logger,
    and other server configuration needed for handling API requests.
    Automatically detects pure diffusion mode (single diffusion stage) and
    handles it appropriately.

    Args:
        engine_client: Engine client instance (AsyncOmni)
        state: FastAPI application state object to initialize
        args: Parsed command-line arguments
    """
    # Get vllm_config from engine_client (following 0.14.0 pattern)
    vllm_config = await _get_vllm_config(engine_client)

    # Detect if it's pure Diffusion mode (single stage and is Diffusion)
    is_pure_diffusion = False
    if hasattr(engine_client, "stage_configs") and engine_client.stage_configs:
        stage_configs = engine_client.stage_configs
        if len(stage_configs) == 1:
            stage_type = get_stage_type(stage_configs[0])
            if stage_type == "diffusion":
                is_pure_diffusion = True
                logger.info("Detected pure diffusion mode (single diffusion stage)")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [BaseModelPath(name=name, model_path=args.model) for name in served_model_names]
    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.args = args

    # For omni models
    state.stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
    model_name = served_model_names[0] if served_model_names else args.model

    # Pure Diffusion mode: use simplified initialization logic
    if is_pure_diffusion:
        state.vllm_config = None
        state.diffusion_engine = engine_client
        state.openai_serving_models = _DiffusionServingModels(base_model_paths)
        # OMNI: tokenization endpoints are not supported in pure diffusion mode.
        state.openai_serving_tokenization = None

        # Use for_diffusion method to create chat handler
        state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
        )

        # audio related
        state.openai_serving_speech = None
        state.openai_serving_audio_generate = OmniOpenAIServingAudioGenerate.for_diffusion(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            model_name=model_name,
        )

        # video related
        diffusion_stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )

        state.openai_serving_speech = OmniOpenAIServingSpeech.for_diffusion(
            diffusion_engine=engine_client,
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )
        state.openai_streaming_speech = None
        state.openai_streaming_video = None

        state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
        state.server_load_metrics = 0
        logger.info("Pure diffusion API server initialized for model: %s", model_name)
        return

    # LLM or multi-stage mode: use standard initialization logic
    if vllm_config is None:
        # Try to get vllm_config from engine_client
        vllm_config = await _get_vllm_config(engine_client)
        if vllm_config is None:
            logger.warning("vllm_config is None, some features may not work correctly")

    state.vllm_config = vllm_config

    # Get supported tasks
    supported_tasks: set[str] = {"generate"}
    if hasattr(engine_client, "get_supported_tasks"):
        supported_tasks = set(await engine_client.get_supported_tasks())
    logger.info("Supported tasks: %s", supported_tasks)

    resolved_chat_template = load_chat_template(args.chat_template)

    if args.tool_server == "demo":
        tool_server: ToolServer | None = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config is not None and vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    # Ensure `input_processor` and `model_config` exist on the engine client
    # for OpenAIServingModels compatibility.
    #
    # vLLM 0.20 dropped the `io_processor` kwarg from OpenAIServingRender and
    # neither `vllm.entrypoints.openai.*` nor `vllm.entrypoints.serve.*` reads
    # `engine_client.io_processor` anymore, so we no longer need to back-fill
    # it here. AsyncOmni still sets `self.io_processor` in its own __init__
    # for any vllm-omni internal callers that rely on it.
    if (
        not hasattr(engine_client, "input_processor")
        or engine_client.input_processor is None
        or not hasattr(engine_client, "model_config")
        or engine_client.model_config is None
    ):
        if vllm_config is not None:
            try:
                from vllm.v1.engine.input_processor import InputProcessor

                tokenizer = await engine_client.get_tokenizer()
                if tokenizer is not None:
                    if not hasattr(engine_client, "input_processor") or engine_client.input_processor is None:
                        engine_client.input_processor = InputProcessor(
                            vllm_config=vllm_config,
                        )
                        logger.info("Initialized input_processor for AsyncOmni")

                    if not hasattr(engine_client, "model_config") or engine_client.model_config is None:
                        engine_client.model_config = vllm_config.model_config
                        logger.info("Initialized model_config for AsyncOmni")
                else:
                    logger.warning("Cannot initialize processors: tokenizer is None. OpenAIServingModels may fail.")
            except Exception as e:
                logger.warning(
                    "Failed to initialize processors for AsyncOmni: %s. OpenAIServingModels may fail.",
                    e,
                )
        else:
            logger.warning("Cannot initialize processors: vllm_config is None. OpenAIServingModels may fail.")

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()

    # NOTE: kept aligned with vllm 0.20 `init_app_state`:
    # - dropped the `io_processor` kwarg (no longer accepted by 0.20);
    #   io_processor stays on `engine_client` and downstream serving classes
    #   read it from there.
    # - pass `reasoning_parser` so render-time `adjust_request` runs for
    #   reasoning models (matches `vllm.entrypoints.openai.api_server`).
    state.openai_serving_render = OpenAIServingRender(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        model_registry=state.openai_serving_models.registry,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_server=tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_chat = (
        OmniOpenAIServingChat(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
            trust_request_chat_template=args.trust_request_chat_template,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
            enable_log_deltas=args.enable_log_deltas,
        )
        if "generate" in supported_tasks
        else None
    )
    # Warm up chat template processing to avoid first-request latency
    if state.openai_serving_chat is not None:
        state.openai_serving_chat.warmup()

    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_pooling = (
        ServingPooling(
            engine_client,
            state.openai_serving_models,
            supported_tasks=tuple(supported_tasks),
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if any(task in POOLING_TASKS for task in supported_tasks)
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if "embed" in supported_tasks
        else None
    )
    state.openai_serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
        )
        if "classify" in supported_tasks
        else None
    )
    state.openai_serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            score_template=resolved_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if any(t in supported_tasks for t in ("embed", "score", "token_embed"))
        else None
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        state.openai_serving_models,
        state.openai_serving_render,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )
    state.openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.openai_serving_translation = (
        OpenAIServingTranslation(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.anthropic_serving_messages = (
        AnthropicServingMessages(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
        )
        if "generate" in supported_tasks
        else None
    )

    state.openai_serving_speech = OmniOpenAIServingSpeech(
        engine_client, state.openai_serving_models, request_logger=request_logger, model_name=model_name
    )

    state.openai_serving_audio_generate = OmniOpenAIServingAudioGenerate(
        engine_client, state.openai_serving_models, request_logger=request_logger, model_name=model_name
    )

    state.openai_streaming_speech = OmniStreamingSpeechHandler(
        speech_service=state.openai_serving_speech,
    )
    state.openai_streaming_video = (
        OmniStreamingVideoHandler(
            chat_service=state.openai_serving_chat,
            engine_client=engine_client,
        )
        if state.openai_serving_chat is not None
        else None
    )
    state.openai_serving_realtime = OpenAIServingRealtime(
        engine_client=engine_client,
        models=state.openai_serving_models,
        request_logger=request_logger,
    )

    state.openai_serving_video = OmniOpenAIServingVideo(
        engine_client,
        model_name=served_model_names[0] if served_model_names else None,
        stage_configs=state.stage_configs,
    )

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0
    state.sleeping_stages = set()


def Omnivideo(request: Request) -> OmniOpenAIServingVideo | None:
    return request.app.state.openai_serving_video


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def Omnispeech(request: Request) -> OmniOpenAIServingSpeech | None:
    return request.app.state.openai_serving_speech


def OmniAudioGenerate(request: Request) -> OmniOpenAIServingAudioGenerate | None:
    return getattr(request.app.state, "openai_serving_audio_generate", None)


class OmniPrefetchSessionRequest(BaseModel):
    """Request-level warm prefetch prototype for interactive KV reload.

    This endpoint intentionally admits a short text-only request. It is a
    control-plane wrapper around the P3c warm-request experiment. P4 adds an
    admission gate and low-priority scheduling so the warm request behaves like
    scheduler-owned background work instead of competing as an ordinary turn.
    """

    messages: list[dict[str, Any]]
    model: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    committed_prefix_hash: str | None = None
    alias_prefix_hashes: list[str] | None = None
    prefix_token_estimate: int | None = Field(default=None, ge=0)
    lead_ms: float | None = None
    max_tokens: int = Field(default=1, ge=1, le=16)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    admission_control: bool = True
    prefetch_priority: int | None = None
    stream_options: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


_OMNI_EVICTION_METADATA_FIELDS = (
    "omni_round_index",
    "omni_trace_timestamp",
    "omni_query_length",
    "omni_response_length",
    "omni_history_tokens_before",
    "omni_history_tokens_after",
    "omni_prev_gap_sec",
    "omni_expected_next_gap_sec",
    "omni_expected_next_use_ts",
    "omni_audio_total_sec",
    "omni_audio_remaining_at_finish_sec",
    "omni_audio_reply_delay_sec",
    "omni_is_terminal_turn",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_policy_enabled(name: str) -> bool:
    raw = os.environ.get(name, "")
    return str(raw).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
        "none",
        "lru",
        "actual_lru",
    }


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default


def _env_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values: list[int] = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(max(0, int(item)))
        except ValueError:
            logger.warning("Invalid %s item %r; using %s", name, item, default)
            return default
    return tuple(sorted(set(values))) or default


def _safe_qsize(queue_like: Any) -> int:
    try:
        return int(queue_like.qsize())
    except Exception:
        return 0


def _scheduler_policy_name_from_config(vllm_config: Any) -> str:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    policy = getattr(scheduler_config, "policy", None)
    value = getattr(policy, "value", None)
    if value is None:
        value = str(policy or "")
    return str(value).lower()


def _scheduler_policy_name(state: State, stage_index: int = 0) -> str:
    engine_client = getattr(state, "engine_client", None)
    engine = getattr(engine_client, "engine", None)
    stage_pools = list(getattr(engine, "stage_pools", []) or [])
    if 0 <= stage_index < len(stage_pools):
        policy = _scheduler_policy_name_from_config(
            getattr(stage_pools[stage_index], "stage_vllm_config", None)
        )
        if policy:
            return policy

    # Fallback for single-stage/non-Omni serving paths.
    return _scheduler_policy_name_from_config(getattr(state, "vllm_config", None))


def _build_omni_load_snapshot(state: State) -> dict[str, Any]:
    """Best-effort load snapshot used by interactive prefetch admission.

    The snapshot intentionally reads only cheap front-end/orchestrator-owned
    structures. It is an admission hint rather than an exact scheduler dump.
    """

    engine_client = getattr(state, "engine_client", None)
    engine = getattr(engine_client, "engine", None)
    stage_pools = list(getattr(engine, "stage_pools", []) or [])

    request_queue = getattr(getattr(engine, "request_queue", None), "sync_q", None)
    output_queue = getattr(getattr(engine, "output_queue", None), "sync_q", None)
    rpc_queue = getattr(getattr(engine, "rpc_output_queue", None), "sync_q", None)

    stages: list[dict[str, Any]] = []
    max_audio_active_per_replica = 0.0
    max_stage0_active_per_replica = 0.0
    total_active_bindings = 0
    for index, pool in enumerate(stage_pools):
        bindings = dict(getattr(pool, "_request_bindings", {}) or {})
        activity_fn = getattr(pool, "active_request_snapshot", None)
        use_stage_activity = bool(getattr(pool, "_stage_active_load_enabled", False)) and callable(activity_fn)
        track_activity = callable(activity_fn) and getattr(pool, "_track_stage_activity", False)
        outstanding = activity_fn() if track_activity else None
        active_requests = outstanding if use_stage_activity else bindings
        num_replicas = max(1, int(getattr(pool, "num_replicas", 1) or 1))
        per_replica = [0 for _ in range(num_replicas)]
        for replica_id in active_requests.values():
            try:
                ridx = int(replica_id)
            except Exception:
                ridx = 0
            if 0 <= ridx < num_replicas:
                per_replica[ridx] += 1
        binding_per_replica = [0 for _ in range(num_replicas)] if use_stage_activity else per_replica
        if use_stage_activity:
            for replica_id in bindings.values():
                if isinstance(replica_id, int) and 0 <= replica_id < num_replicas:
                    binding_per_replica[replica_id] += 1
        outstanding_per_replica = None
        if outstanding is not None:
            outstanding_per_replica = [0 for _ in range(num_replicas)]
            for replica_id in outstanding.values():
                if isinstance(replica_id, int) and 0 <= replica_id < num_replicas:
                    outstanding_per_replica[replica_id] += 1
        active = len(active_requests)
        total_active_bindings += len(bindings)
        active_per_replica = float(max(per_replica) if per_replica else 0)
        avg_active_per_replica = float(active) / float(num_replicas)
        stage_id = int(getattr(pool, "stage_id", index))
        stage_type = getattr(pool, "stage_type", None)
        scheduler_policy = _scheduler_policy_name_from_config(
            getattr(pool, "stage_vllm_config", None)
        )
        row = {
            "stage_index": index,
            "stage_id": stage_id,
            "stage_type": stage_type,
            "scheduler_policy": scheduler_policy,
            "num_replicas": num_replicas,
            "active_bindings": len(bindings),
            "outstanding_stage_requests": len(outstanding) if outstanding is not None else None,
            "binding_by_replica": binding_per_replica,
            "outstanding_by_replica": outstanding_per_replica,
            "pressure_source": "stage_outstanding" if use_stage_activity else "route_bindings",
            "pressure_semantics": "submitted_stage_work" if use_stage_activity else "request_lifetime_bindings",
            "GPU_running_count_confirmed": None,
            "target_replica_admission_confirmed": False,
            "active_by_replica": per_replica,
            "active_per_replica": active_per_replica,
            "avg_active_per_replica": avg_active_per_replica,
        }
        stages.append(row)
        if stage_id == 0 or index == 0:
            max_stage0_active_per_replica = max(
                max_stage0_active_per_replica,
                active_per_replica,
            )
        else:
            max_audio_active_per_replica = max(
                max_audio_active_per_replica,
                active_per_replica,
            )

    return {
        "active_frontend_requests": len(getattr(engine_client, "request_states", {}) or {}),
        "orchestrator_request_qsize": _safe_qsize(request_queue),
        "orchestrator_output_qsize": _safe_qsize(output_queue),
        "orchestrator_rpc_qsize": _safe_qsize(rpc_queue),
        "total_active_bindings": total_active_bindings,
        "max_stage0_active_per_replica": max_stage0_active_per_replica,
        "max_audio_active_per_replica": max_audio_active_per_replica,
        "stages": stages,
        "scheduler_policy": _scheduler_policy_name(state),
    }


class OmniRuntimeStateBus:
    """Low-overhead state bridge for joint Talker/Thinker scheduling.

    This is deliberately an API-process monitor first: it records coarse
    live/prefetch transition events and recent stage-pressure snapshots. The
    admission controller can then gate background prefetch with fresh runtime
    context without requiring a deep scheduler refactor in the first step.
    """

    def __init__(self) -> None:
        self.enabled = _env_bool("OMNI_RUNTIME_MONITOR", False)
        self.trace_enabled = _env_bool("OMNI_RUNTIME_MONITOR_TRACE", False)
        self.trace_dir = os.environ.get(
            "OMNI_RUNTIME_MONITOR_DIR",
            "/tmp/omni_runtime_monitor",
        )
        self.ttl_s = max(1.0, _env_float("OMNI_RUNTIME_MONITOR_TTL_SEC", 120.0))
        self.max_events = max(64, _env_int("OMNI_RUNTIME_MONITOR_MAX_EVENTS", 4096))
        self.live_window_ms = max(
            0.0,
            _env_float("OMNI_RUNTIME_MONITOR_LIVE_WINDOW_MS", 1500.0),
        )
        self.audio_backlog_threshold = _env_float(
            "OMNI_RUNTIME_MONITOR_AUDIO_BACKLOG_THRESHOLD",
            1.0,
        )
        self.stage0_busy_threshold = _env_float(
            "OMNI_RUNTIME_MONITOR_STAGE0_BUSY_THRESHOLD",
            3.0,
        )
        self._lock = asyncio.Lock()
        self._seq = 0
        self._live_events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._abort_events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._prefetch_events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._last_audio_backlog_monotonic: float | None = None
        self._last_stage0_busy_monotonic: float | None = None
        self._last_snapshot: dict[str, Any] | None = None

    def _trace_path(self) -> str:
        return os.path.join(
            self.trace_dir,
            f"runtime_monitor.{os.getpid()}.jsonl",
        )

    @staticmethod
    def _matches(
        event: dict[str, Any],
        *,
        session_id: str | None,
        prefix_hash: str | None,
    ) -> tuple[bool, bool]:
        same_session = bool(session_id and event.get("session_id") == session_id)
        same_prefix = bool(
            same_session
            and prefix_hash
            and event.get("committed_prefix_hash") == prefix_hash
        )
        return same_session, same_prefix

    def _purge_locked(self, now: float) -> None:
        cutoff = now - self.ttl_s
        for events in (self._live_events, self._abort_events, self._prefetch_events):
            while events and float(events[0].get("monotonic", 0.0)) < cutoff:
                events.popleft()

    def _record_pressure_locked(self, snapshot: dict[str, Any] | None, now: float) -> None:
        if not isinstance(snapshot, dict):
            return
        self._last_snapshot = {
            "active_frontend_requests": snapshot.get("active_frontend_requests"),
            "orchestrator_request_qsize": snapshot.get("orchestrator_request_qsize"),
            "max_stage0_active_per_replica": snapshot.get(
                "max_stage0_active_per_replica"
            ),
            "max_audio_active_per_replica": snapshot.get(
                "max_audio_active_per_replica"
            ),
            "total_active_bindings": snapshot.get("total_active_bindings"),
        }
        if (
            float(snapshot.get("max_audio_active_per_replica") or 0.0)
            > self.audio_backlog_threshold
        ):
            self._last_audio_backlog_monotonic = now
        if (
            float(snapshot.get("max_stage0_active_per_replica") or 0.0)
            > self.stage0_busy_threshold
        ):
            self._last_stage0_busy_monotonic = now

    async def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if not (self.enabled and self.trace_enabled):
            return
        record = {
            "seq": payload.get("seq"),
            "kind": kind,
            "time_wall": time.time(),
            "time_monotonic": time.monotonic(),
            **payload,
        }
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            with open(self._trace_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("OMNI runtime monitor failed to write trace")

    async def record_live_arrival(
        self,
        *,
        request_id: str | None,
        session_id: str | None,
        prefix_hash: str | None,
        prefix_token_estimate: int | None,
        kv_policy: dict[str, Any] | None,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        now = time.monotonic()
        event = {
            "kind": "live_arrival",
            "monotonic": now,
            "request_id": request_id,
            "session_id": session_id,
            "committed_prefix_hash": prefix_hash,
            "prefix_token_estimate": prefix_token_estimate,
            "kv_lookup_state": (kv_policy or {}).get("lookup_state"),
            "kv_prefetch_hit_used": (kv_policy or {}).get("prefetch_hit_used"),
            "kv_skip_offload": (kv_policy or {}).get("skip_offload"),
        }
        async with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._purge_locked(now)
            self._record_pressure_locked(snapshot, now)
            self._live_events.append(event)
            stats = self._stats_locked(now)
        await self._emit("live_arrival", {**event, "stats": stats})
        return stats

    async def record_live_abort(
        self,
        *,
        session_id: str | None,
        prefix_hash: str | None,
        abort: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        now = time.monotonic()
        event = {
            "kind": "live_abort",
            "monotonic": now,
            "session_id": session_id,
            "committed_prefix_hash": prefix_hash,
            "aborted": int((abort or {}).get("aborted") or 0),
            "request_ids": (abort or {}).get("request_ids") or [],
        }
        async with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._purge_locked(now)
            self._abort_events.append(event)
            stats = self._stats_locked(now)
        await self._emit("live_abort", {**event, "stats": stats})
        return stats

    async def record_prefetch_decision(
        self,
        *,
        request_id: str,
        session_id: str | None,
        prefix_hash: str | None,
        admitted: bool,
        admission: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        now = time.monotonic()
        event = {
            "kind": "prefetch_decision",
            "monotonic": now,
            "request_id": request_id,
            "session_id": session_id,
            "committed_prefix_hash": prefix_hash,
            "admitted": bool(admitted),
            "reason": (admission or {}).get("reason"),
            "joint_reason": ((admission or {}).get("joint_scheduler") or {}).get(
                "reason"
            ),
            "lead_ms": (admission or {}).get("lead_ms"),
            "prefix_token_estimate": (admission or {}).get("prefix_token_estimate"),
        }
        snapshot = (admission or {}).get("snapshot")
        async with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._purge_locked(now)
            self._record_pressure_locked(snapshot, now)
            self._prefetch_events.append(event)
            stats = self._stats_locked(now)
        await self._emit("prefetch_decision", {**event, "stats": stats})
        return stats

    async def record_prefetch_start(
        self,
        *,
        request_id: str,
        session_id: str | None,
        prefix_hash: str | None,
        admission: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        now = time.monotonic()
        event = {
            "kind": "prefetch_start",
            "monotonic": now,
            "request_id": request_id,
            "session_id": session_id,
            "committed_prefix_hash": prefix_hash,
            "reason": (admission or {}).get("reason"),
        }
        async with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._purge_locked(now)
            self._prefetch_events.append(event)
            stats = self._stats_locked(now)
        await self._emit("prefetch_start", {**event, "stats": stats})
        return stats

    async def record_prefetch_finish(
        self,
        *,
        request_id: str,
        session_id: str | None,
        prefix_hash: str | None,
        terminal_status: str,
        wall_s: float | None,
        controller: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        now = time.monotonic()
        event = {
            "kind": "prefetch_finish",
            "monotonic": now,
            "request_id": request_id,
            "session_id": session_id,
            "committed_prefix_hash": prefix_hash,
            "terminal_status": terminal_status,
            "wall_s": wall_s,
            "controller": controller,
        }
        async with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._purge_locked(now)
            self._prefetch_events.append(event)
            stats = self._stats_locked(now)
        await self._emit("prefetch_finish", {**event, "stats": stats})
        return stats

    async def policy_context(
        self,
        *,
        session_id: str | None,
        prefix_hash: str | None,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        now = time.monotonic()
        async with self._lock:
            self._purge_locked(now)
            self._record_pressure_locked(snapshot, now)
            window_cutoff = now - self.live_window_ms / 1000.0
            recent_live = [
                event
                for event in self._live_events
                if float(event.get("monotonic", 0.0)) >= window_cutoff
            ]
            same_session_live = 0
            same_prefix_live = 0
            for event in recent_live:
                same_session, same_prefix = self._matches(
                    event,
                    session_id=session_id,
                    prefix_hash=prefix_hash,
                )
                same_session_live += int(same_session)
                same_prefix_live += int(same_prefix)
            same_session_aborts = [
                event
                for event in self._abort_events
                if self._matches(
                    event,
                    session_id=session_id,
                    prefix_hash=prefix_hash,
                )[0]
            ]
            same_prefix_aborts = [
                event
                for event in self._abort_events
                if self._matches(
                    event,
                    session_id=session_id,
                    prefix_hash=prefix_hash,
                )[1]
            ]
            last_abort = same_prefix_aborts[-1:] or same_session_aborts[-1:]
            recent_abort_age_ms = (
                (now - float(last_abort[0].get("monotonic", now))) * 1000.0
                if last_abort
                else None
            )
            audio_age_ms = (
                (now - self._last_audio_backlog_monotonic) * 1000.0
                if self._last_audio_backlog_monotonic is not None
                else None
            )
            stage0_age_ms = (
                (now - self._last_stage0_busy_monotonic) * 1000.0
                if self._last_stage0_busy_monotonic is not None
                else None
            )
            out = {
                "enabled": True,
                "live_window_ms": self.live_window_ms,
                "recent_live_count": len(recent_live),
                "same_session_recent_live": same_session_live,
                "same_prefix_recent_live": same_prefix_live,
                "same_session_abort_count": len(same_session_aborts),
                "same_prefix_abort_count": len(same_prefix_aborts),
                "recent_abort_age_ms": recent_abort_age_ms,
                "audio_backlog_age_ms": audio_age_ms,
                "stage0_busy_age_ms": stage0_age_ms,
                "last_snapshot": self._last_snapshot,
                "trace_enabled": self.trace_enabled,
                "trace_path": self._trace_path() if self.trace_enabled else None,
                "stats": self._stats_locked(now),
            }
        return out

    def _stats_locked(self, now: float) -> dict[str, Any]:
        window_cutoff = now - self.live_window_ms / 1000.0
        return {
            "enabled": self.enabled,
            "trace_enabled": self.trace_enabled,
            "live_events": len(self._live_events),
            "abort_events": len(self._abort_events),
            "prefetch_events": len(self._prefetch_events),
            "recent_live_count": sum(
                1
                for event in self._live_events
                if float(event.get("monotonic", 0.0)) >= window_cutoff
            ),
            "audio_backlog_age_ms": (
                (now - self._last_audio_backlog_monotonic) * 1000.0
                if self._last_audio_backlog_monotonic is not None
                else None
            ),
            "stage0_busy_age_ms": (
                (now - self._last_stage0_busy_monotonic) * 1000.0
                if self._last_stage0_busy_monotonic is not None
                else None
            ),
        }


def _get_runtime_monitor(state: State) -> OmniRuntimeStateBus:
    monitor = getattr(state, "omni_runtime_monitor", None)
    if monitor is None:
        monitor = OmniRuntimeStateBus()
        state.omni_runtime_monitor = monitor
    return monitor


class OmniPrefetchAdmissionController:
    """Small admission controller for interactive history prefetch.

    The policy is intentionally conservative. It admits prefetch only when
    VAD lead time is long enough and current stage load leaves room for
    background KV reload. Native priority scheduling is still used after
    admission, so an admitted prefetch yields to live turns.
    """

    def __init__(self) -> None:
        self.enabled = _env_bool("OMNI_PREFETCH_ADMISSION", False)
        self.min_lead_ms = _env_float("OMNI_PREFETCH_MIN_LEAD_MS", 600.0)
        self.max_stage0_active_per_replica = _env_float(
            "OMNI_PREFETCH_MAX_STAGE0_ACTIVE_PER_REPLICA",
            4.0,
        )
        self.max_audio_active_per_replica = _env_float(
            "OMNI_PREFETCH_MAX_AUDIO_ACTIVE_PER_REPLICA",
            2.0,
        )
        self.max_request_qsize = _env_int("OMNI_PREFETCH_MAX_REQUEST_QSIZE", 8)
        self.max_inflight = max(1, _env_int("OMNI_PREFETCH_SERVER_MAX_INFLIGHT", 1))
        self.default_priority = _env_int(
            "OMNI_PREFETCH_PRIORITY",
            100 if self.enabled else 0,
        )
        self.deadline_aware = _env_bool("OMNI_PREFETCH_DEADLINE_AWARE", False)
        self.deadline_guard_ms = _env_float(
            "OMNI_PREFETCH_DEADLINE_GUARD_MS",
            500.0,
        )
        self.deadline_initial_estimate_ms = max(
            1.0,
            _env_float("OMNI_PREFETCH_DEADLINE_INITIAL_ESTIMATE_MS", 250.0),
        )
        self.deadline_alpha = min(
            1.0,
            max(0.01, _env_float("OMNI_PREFETCH_DEADLINE_EWMA_ALPHA", 0.25)),
        )
        self.deadline_inflight_penalty_ms = max(
            0.0,
            _env_float("OMNI_PREFETCH_DEADLINE_INFLIGHT_PENALTY_MS", 250.0),
        )
        self.deadline_stage0_penalty_ms = max(
            0.0,
            _env_float("OMNI_PREFETCH_DEADLINE_STAGE0_PENALTY_MS", 0.0),
        )
        self.deadline_prefix_buckets = _env_int_list(
            "OMNI_PREFETCH_DEADLINE_PREFIX_BUCKETS",
            (256, 512, 1024, 2048, 4096, 8192, 16384),
        )
        self.deadline_prefix_ms_per_1k = max(
            0.0,
            _env_float("OMNI_PREFETCH_DEADLINE_PREFIX_MS_PER_1K", 25.0),
        )
        self.deadline_min_bucket_samples = max(
            1,
            _env_int("OMNI_PREFETCH_DEADLINE_MIN_BUCKET_SAMPLES", 2),
        )
        self.abort_on_live = _env_bool("OMNI_PREFETCH_ABORT_ON_LIVE", False)
        self.abort_scope = os.environ.get(
            "OMNI_PREFETCH_ABORT_SCOPE",
            "prefix",
        ).strip().lower()
        self.shadow_enabled = _env_bool("OMNI_PREFETCH_SHADOW_POLICY", True)
        self.shadow_min_prefix_tokens = max(
            0,
            _env_int("OMNI_PREFETCH_SHADOW_MIN_PREFIX_TOKENS", 16),
        )
        self.joint_scheduler_enabled = _env_bool("OMNI_JOINT_SCHEDULER", False)
        self.joint_scheduler_shadow = _env_bool("OMNI_JOINT_SCHEDULER_SHADOW", True)
        self.joint_wait_ms = max(0.0, _env_float("OMNI_JOINT_SCHEDULER_WAIT_MS", 0.0))
        self.joint_wait_poll_ms = max(
            1.0,
            _env_float("OMNI_JOINT_SCHEDULER_WAIT_POLL_MS", 25.0),
        )
        self.joint_active_live_limit = _env_int("OMNI_JOINT_ACTIVE_LIVE_LIMIT", -1)
        self.joint_same_session_live_limit = _env_int(
            "OMNI_JOINT_SAME_SESSION_LIVE_LIMIT",
            -1,
        )
        self.joint_recent_abort_ms = max(
            0.0,
            _env_float("OMNI_JOINT_RECENT_ABORT_MS", 0.0),
        )
        self.joint_audio_backlog_hold_ms = max(
            0.0,
            _env_float("OMNI_JOINT_AUDIO_BACKLOG_HOLD_MS", 0.0),
        )
        self.joint_stage0_busy_hold_ms = max(
            0.0,
            _env_float("OMNI_JOINT_STAGE0_BUSY_HOLD_MS", 0.0),
        )
        wait_reasons_raw = os.environ.get(
            "OMNI_JOINT_SCHEDULER_WAIT_REASONS",
            "high_thinker_pressure,high_audio_backlog,prefetch_inflight_limit,high_request_queue,monitor_recent_live,monitor_same_session_live,monitor_recent_abort,monitor_audio_backlog_hold,monitor_stage0_busy_hold",
        )
        self.joint_wait_reasons = {
            item.strip()
            for item in wait_reasons_raw.replace(";", ",").split(",")
            if item.strip()
        }
        self._lock = asyncio.Lock()
        self._inflight = 0
        self._accepted = 0
        self._skipped = 0
        self._joint_admitted = 0
        self._joint_skipped = 0
        self._joint_waits = 0
        self._joint_wait_admitted = 0
        self._joint_wait_timeout = 0
        self._active: dict[str, dict[str, Any]] = {}
        self._live_abort_calls = 0
        self._live_abort_requests = 0
        self._prefetch_wall_ewma_ms = self.deadline_initial_estimate_ms
        self._prefetch_wall_samples = 0
        self._deadline_bucket_stats: dict[str, dict[str, float | int]] = {}

    def _prefix_token_bucket(self, prefix_tokens: int | None) -> str:
        if prefix_tokens is None:
            return "unknown"
        tokens = max(0, int(prefix_tokens))
        lower = 0
        for upper in self.deadline_prefix_buckets:
            if tokens <= upper:
                return f"{lower}-{upper}"
            lower = upper + 1
        return f"{lower}+"

    @staticmethod
    def _load_bucket(value: float | int | None) -> str:
        v = float(value or 0.0)
        if v <= 0:
            return "0"
        if v <= 1:
            return "1"
        if v <= 2:
            return "2"
        if v <= 4:
            return "3-4"
        return "5+"

    def _deadline_bucket_key(
        self,
        *,
        prefix_tokens: int | None,
        snapshot: dict[str, Any],
    ) -> str:
        return (
            f"p={self._prefix_token_bucket(prefix_tokens)}"
            f"|s0={self._load_bucket(snapshot.get('max_stage0_active_per_replica'))}"
            f"|aud={self._load_bucket(snapshot.get('max_audio_active_per_replica'))}"
            f"|q={self._load_bucket(snapshot.get('orchestrator_request_qsize'))}"
        )

    def _bucket_initial_estimate_ms(self, prefix_tokens: int | None) -> float:
        if prefix_tokens is None:
            return self.deadline_initial_estimate_ms
        return (
            self.deadline_initial_estimate_ms
            + max(0.0, float(prefix_tokens)) / 1000.0
            * self.deadline_prefix_ms_per_1k
        )

    def _bucket_stat(self, key: str) -> dict[str, float | int] | None:
        stat = self._deadline_bucket_stats.get(key)
        if stat is None:
            return None
        if int(stat.get("samples", 0) or 0) < self.deadline_min_bucket_samples:
            return None
        return stat

    def _deadline_base_estimate_ms(
        self,
        *,
        key: str,
        prefix_bucket_key: str,
        prefix_tokens: int | None,
    ) -> tuple[float, str, int]:
        for candidate_key, source in (
            (key, "load_prefix_bucket"),
            (prefix_bucket_key, "prefix_bucket"),
        ):
            stat = self._bucket_stat(candidate_key)
            if stat is not None:
                return (
                    float(stat.get("ewma_ms") or self.deadline_initial_estimate_ms),
                    source,
                    int(stat.get("samples", 0) or 0),
                )
        if self._prefetch_wall_samples > 0:
            return (
                float(self._prefetch_wall_ewma_ms),
                "global_ewma",
                int(self._prefetch_wall_samples),
            )
        return (
            self._bucket_initial_estimate_ms(prefix_tokens),
            "prefix_initial",
            0,
        )

    async def _reject_admission(
        self,
        decision: dict[str, Any],
        reason: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Record a non-reserving rejection without changing admission order."""
        decision.update({"admitted": False, "reason": reason})
        async with self._lock:
            self._skipped += 1
            decision["controller"] = self._controller_stats()
        return False, decision

    async def decide(
        self,
        request: OmniPrefetchSessionRequest,
        state: State,
    ) -> tuple[bool, dict[str, Any]]:
        snapshot = _build_omni_load_snapshot(state)
        lead_ms = request.lead_ms
        prefix_tokens = request.prefix_token_estimate
        deadline_ctx = self._deadline_context(prefix_tokens, snapshot)
        monitor_ctx = await _get_runtime_monitor(state).policy_context(
            session_id=request.session_id,
            prefix_hash=request.committed_prefix_hash,
            snapshot=snapshot,
        )
        decision: dict[str, Any] = {
            "enabled": self.enabled and request.admission_control,
            "inflight_reserved": False,
            "lead_ms": lead_ms,
            "prefix_token_estimate": prefix_tokens,
            "min_lead_ms": self.min_lead_ms,
            "max_stage0_active_per_replica": self.max_stage0_active_per_replica,
            "max_audio_active_per_replica": self.max_audio_active_per_replica,
            "max_request_qsize": self.max_request_qsize,
            "max_inflight": self.max_inflight,
            "deadline_aware": self.deadline_aware,
            "deadline_guard_ms": self.deadline_guard_ms,
            "deadline_estimate_ms": deadline_ctx["estimate_ms"],
            "deadline_bucket": deadline_ctx,
            "snapshot": snapshot,
            "runtime_monitor": monitor_ctx,
        }
        decision["shadow_policy"] = self._shadow_joint_policy(
            request,
            snapshot,
            deadline_ctx,
            monitor_ctx,
        )
        joint_policy = decision["shadow_policy"]
        joint_enforcing = (
            self.joint_scheduler_enabled
            and not self.joint_scheduler_shadow
            and request.admission_control
        )
        decision["joint_scheduler"] = {
            "enabled": self.joint_scheduler_enabled,
            "shadow": self.joint_scheduler_shadow,
            "enforcing": joint_enforcing,
            "policy": joint_policy.get("policy"),
            "action": joint_policy.get("action"),
            "reason": joint_policy.get("reason"),
        }
        decision["joint_enforced"] = joint_enforcing
        if joint_enforcing:
            (
                joint_policy,
                wait_info,
                snapshot,
                deadline_ctx,
                monitor_ctx,
            ) = await self._maybe_wait_for_joint_admit(
                request,
                state,
                joint_policy,
            )
            if wait_info is not None:
                decision["joint_wait"] = wait_info
                decision["shadow_policy"] = joint_policy
                decision["snapshot"] = snapshot
                decision["deadline_bucket"] = deadline_ctx
                decision["runtime_monitor"] = monitor_ctx
                decision["deadline_estimate_ms"] = deadline_ctx["estimate_ms"]
                decision["lead_ms"] = wait_info.get("remaining_lead_ms", lead_ms)
                lead_ms = decision["lead_ms"]
                prefix_tokens = request.prefix_token_estimate
            decision["joint_scheduler"].update(
                {
                    "action": joint_policy.get("action"),
                    "reason": joint_policy.get("reason"),
                }
            )
        if joint_enforcing and joint_policy.get("action") != "admit":
            decision.update(
                {
                    "admitted": False,
                    "reason": f"joint_{joint_policy.get('reason', 'rejected')}",
                    "joint_enforced": True,
                }
            )
            async with self._lock:
                self._skipped += 1
                self._joint_skipped += 1
                decision["controller"] = self._controller_stats()
            return False, decision

        legacy_enabled = self.enabled or joint_enforcing
        if not legacy_enabled or not request.admission_control:
            decision.update({"admitted": True, "reason": "admission_disabled"})
            return True, decision
        if lead_ms is None or lead_ms < self.min_lead_ms:
            return await self._reject_admission(decision, "vad_lead_too_short")
        if snapshot["orchestrator_request_qsize"] > self.max_request_qsize:
            return await self._reject_admission(decision, "orchestrator_queue_busy")
        if snapshot["max_stage0_active_per_replica"] > self.max_stage0_active_per_replica:
            return await self._reject_admission(decision, "thinker_pressure")
        if snapshot["max_audio_active_per_replica"] > self.max_audio_active_per_replica:
            return await self._reject_admission(decision, "audio_backlog")
        if self.deadline_aware:
            deadline = self._deadline_decision(lead_ms, deadline_ctx)
            decision["deadline"] = deadline
            if not deadline["admitted"]:
                return await self._reject_admission(decision, "prefetch_deadline_too_tight")
        async with self._lock:
            if self._inflight >= self.max_inflight:
                self._skipped += 1
                decision.update(
                    {
                        "admitted": False,
                        "reason": "prefetch_inflight_limit",
                        "controller": self._controller_stats(),
                    }
                )
                return False, decision
            self._inflight += 1
            decision["inflight_reserved"] = True
            self._accepted += 1
            if joint_enforcing:
                self._joint_admitted += 1
            decision.update(
                {
                    "admitted": True,
                    "reason": "admitted",
                    "controller": self._controller_stats(),
                }
            )
            return True, decision

    def _deadline_context(
        self,
        prefix_tokens: int | None,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        key = self._deadline_bucket_key(prefix_tokens=prefix_tokens, snapshot=snapshot)
        prefix_bucket = self._prefix_token_bucket(prefix_tokens)
        prefix_bucket_key = f"p={prefix_bucket}"
        base_ms, source, samples = self._deadline_base_estimate_ms(
            key=key,
            prefix_bucket_key=prefix_bucket_key,
            prefix_tokens=prefix_tokens,
        )
        estimate_ms = (
            float(base_ms)
            + float(self._inflight) * self.deadline_inflight_penalty_ms
            + float(snapshot.get("max_stage0_active_per_replica") or 0.0)
            * self.deadline_stage0_penalty_ms
        )
        return {
            "key": key,
            "prefix_bucket": prefix_bucket,
            "prefix_bucket_key": prefix_bucket_key,
            "source": source,
            "source_samples": samples,
            "base_estimate_ms": base_ms,
            "estimate_ms": estimate_ms,
            "active_inflight": self._inflight,
            "inflight_penalty_ms": self.deadline_inflight_penalty_ms,
            "stage0_penalty_ms": self.deadline_stage0_penalty_ms,
        }

    async def _maybe_wait_for_joint_admit(
        self,
        request: OmniPrefetchSessionRequest,
        state: State,
        initial_policy: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Wait briefly for transient pressure before enforcing a joint defer.

        This is the first scheduler-native bridge: the endpoint remains low
        priority, but it can spend a bounded slice of the VAD lead window
        waiting for stage pressure or prefetch inflight state to improve.
        """

        snapshot = _build_omni_load_snapshot(state)
        deadline_ctx = self._deadline_context(request.prefix_token_estimate, snapshot)
        monitor = _get_runtime_monitor(state)
        monitor_ctx = await monitor.policy_context(
            session_id=request.session_id,
            prefix_hash=request.committed_prefix_hash,
            snapshot=snapshot,
        )
        if initial_policy.get("action") == "admit":
            return initial_policy, None, snapshot, deadline_ctx, monitor_ctx

        reason = str(initial_policy.get("reason") or "")
        lead_ms = request.lead_ms
        if (
            self.joint_wait_ms <= 0
            or reason not in self.joint_wait_reasons
            or lead_ms is None
        ):
            return initial_policy, None, snapshot, deadline_ctx, monitor_ctx

        max_wait_ms = min(
            self.joint_wait_ms,
            max(0.0, float(lead_ms) - self.min_lead_ms),
        )
        if max_wait_ms <= 0:
            return initial_policy, None, snapshot, deadline_ctx, monitor_ctx

        start = time.monotonic()
        attempts = 0
        last_policy = dict(initial_policy)
        last_snapshot = snapshot
        last_deadline_ctx = deadline_ctx
        last_monitor_ctx = monitor_ctx
        final_reason = reason
        admitted_after_wait = False

        async with self._lock:
            self._joint_waits += 1

        while (time.monotonic() - start) * 1000.0 < max_wait_ms:
            await asyncio.sleep(self.joint_wait_poll_ms / 1000.0)
            attempts += 1
            elapsed_ms = (time.monotonic() - start) * 1000.0
            remaining_lead_ms = max(0.0, float(lead_ms) - elapsed_ms)
            wait_request = request.model_copy(
                update={"lead_ms": remaining_lead_ms}
            )
            last_snapshot = _build_omni_load_snapshot(state)
            last_deadline_ctx = self._deadline_context(
                wait_request.prefix_token_estimate,
                last_snapshot,
            )
            last_monitor_ctx = await monitor.policy_context(
                session_id=wait_request.session_id,
                prefix_hash=wait_request.committed_prefix_hash,
                snapshot=last_snapshot,
            )
            last_policy = self._shadow_joint_policy(
                wait_request,
                last_snapshot,
                last_deadline_ctx,
                last_monitor_ctx,
            )
            final_reason = str(last_policy.get("reason") or "")
            if last_policy.get("action") == "admit":
                admitted_after_wait = True
                break
            if final_reason not in self.joint_wait_reasons:
                break

        waited_ms = (time.monotonic() - start) * 1000.0
        if admitted_after_wait:
            async with self._lock:
                self._joint_wait_admitted += 1
        else:
            async with self._lock:
                self._joint_wait_timeout += 1

        wait_info = {
            "enabled": True,
            "initial_action": initial_policy.get("action"),
            "initial_reason": reason,
            "final_action": last_policy.get("action"),
            "final_reason": final_reason,
            "attempts": attempts,
            "waited_ms": waited_ms,
            "max_wait_ms": max_wait_ms,
            "poll_ms": self.joint_wait_poll_ms,
            "admitted_after_wait": admitted_after_wait,
            "remaining_lead_ms": max(0.0, float(lead_ms) - waited_ms),
        }
        return last_policy, wait_info, last_snapshot, last_deadline_ctx, last_monitor_ctx

    def _shadow_joint_policy(
        self,
        request: OmniPrefetchSessionRequest,
        snapshot: dict[str, Any],
        deadline_ctx: dict[str, Any],
        runtime_ctx: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute M2 shadow decision without enforcing it.

        The result is intentionally policy-shaped rather than controller-shaped:
        a future RuntimeStateBus can feed the same predicate with fresher
        scheduler-native state.
        """

        out: dict[str, Any] = {
            "enabled": self.shadow_enabled or self.joint_scheduler_enabled,
            "policy": "m2_joint_shadow_v1",
            "action": "disabled",
            "reason": "shadow_disabled",
            "min_prefix_tokens": self.shadow_min_prefix_tokens,
            "min_lead_ms": self.min_lead_ms,
            "max_stage0_active_per_replica": self.max_stage0_active_per_replica,
            "max_audio_active_per_replica": self.max_audio_active_per_replica,
            "max_request_qsize": self.max_request_qsize,
            "max_inflight": self.max_inflight,
            "deadline_guard_ms": self.deadline_guard_ms,
            "deadline_estimate_ms": deadline_ctx.get("estimate_ms"),
            "deadline_source": deadline_ctx.get("source"),
            "deadline_bucket": deadline_ctx.get("key"),
            "runtime_monitor_enabled": bool((runtime_ctx or {}).get("enabled")),
            "joint_active_live_limit": self.joint_active_live_limit,
            "joint_same_session_live_limit": self.joint_same_session_live_limit,
            "joint_recent_abort_ms": self.joint_recent_abort_ms,
            "joint_audio_backlog_hold_ms": self.joint_audio_backlog_hold_ms,
            "joint_stage0_busy_hold_ms": self.joint_stage0_busy_hold_ms,
        }
        if not (self.shadow_enabled or self.joint_scheduler_enabled):
            return out

        lead_ms = request.lead_ms
        prefix_tokens = request.prefix_token_estimate
        prefix_hash = request.committed_prefix_hash
        out.update(
            {
                "action": "admit",
                "reason": "enough_lead",
                "lead_ms": lead_ms,
                "prefix_token_estimate": prefix_tokens,
                "committed_prefix_hash_present": bool(prefix_hash),
                "snapshot": {
                    "orchestrator_request_qsize": snapshot.get(
                        "orchestrator_request_qsize"
                    ),
                    "max_stage0_active_per_replica": snapshot.get(
                        "max_stage0_active_per_replica"
                    ),
                    "max_audio_active_per_replica": snapshot.get(
                        "max_audio_active_per_replica"
                    ),
                    "active_frontend_requests": snapshot.get(
                        "active_frontend_requests"
                    ),
                    "total_active_bindings": snapshot.get(
                        "total_active_bindings"
                    ),
                    "active_prefetch_inflight": self._inflight,
                },
                "runtime_monitor": {
                    "enabled": bool((runtime_ctx or {}).get("enabled")),
                    "recent_live_count": (runtime_ctx or {}).get(
                        "recent_live_count"
                    ),
                    "same_session_recent_live": (runtime_ctx or {}).get(
                        "same_session_recent_live"
                    ),
                    "same_prefix_recent_live": (runtime_ctx or {}).get(
                        "same_prefix_recent_live"
                    ),
                    "recent_abort_age_ms": (runtime_ctx or {}).get(
                        "recent_abort_age_ms"
                    ),
                    "audio_backlog_age_ms": (runtime_ctx or {}).get(
                        "audio_backlog_age_ms"
                    ),
                    "stage0_busy_age_ms": (runtime_ctx or {}).get(
                        "stage0_busy_age_ms"
                    ),
                },
            }
        )

        if not prefix_hash:
            out.update({"action": "reject", "reason": "stale_prefix"})
            return out
        if prefix_tokens is not None and int(prefix_tokens) < self.shadow_min_prefix_tokens:
            out.update({"action": "reject", "reason": "too_small_prefix"})
            return out
        if lead_ms is None:
            out.update({"action": "reject", "reason": "no_vad_lead"})
            return out
        if float(lead_ms) < self.min_lead_ms:
            out.update({"action": "reject", "reason": "vad_lead_too_short"})
            return out
        if snapshot["orchestrator_request_qsize"] > self.max_request_qsize:
            out.update({"action": "defer", "reason": "high_request_queue"})
            return out
        if snapshot["max_stage0_active_per_replica"] > self.max_stage0_active_per_replica:
            out.update({"action": "defer", "reason": "high_thinker_pressure"})
            return out
        if snapshot["max_audio_active_per_replica"] > self.max_audio_active_per_replica:
            out.update({"action": "defer", "reason": "high_audio_backlog"})
            return out
        if self._inflight >= self.max_inflight:
            out.update({"action": "defer", "reason": "prefetch_inflight_limit"})
            return out
        if (runtime_ctx or {}).get("enabled"):
            if (
                self.joint_active_live_limit >= 0
                and int((runtime_ctx or {}).get("recent_live_count") or 0)
                > self.joint_active_live_limit
            ):
                out.update({"action": "defer", "reason": "monitor_recent_live"})
                return out
            if (
                self.joint_same_session_live_limit >= 0
                and int((runtime_ctx or {}).get("same_session_recent_live") or 0)
                > self.joint_same_session_live_limit
            ):
                out.update(
                    {"action": "defer", "reason": "monitor_same_session_live"}
                )
                return out
            recent_abort_age_ms = (runtime_ctx or {}).get("recent_abort_age_ms")
            if (
                self.joint_recent_abort_ms > 0.0
                and isinstance(recent_abort_age_ms, (int, float))
                and float(recent_abort_age_ms) <= self.joint_recent_abort_ms
            ):
                out.update({"action": "defer", "reason": "monitor_recent_abort"})
                return out
            audio_backlog_age_ms = (runtime_ctx or {}).get("audio_backlog_age_ms")
            if (
                self.joint_audio_backlog_hold_ms > 0.0
                and isinstance(audio_backlog_age_ms, (int, float))
                and float(audio_backlog_age_ms) <= self.joint_audio_backlog_hold_ms
            ):
                out.update(
                    {"action": "defer", "reason": "monitor_audio_backlog_hold"}
                )
                return out
            stage0_busy_age_ms = (runtime_ctx or {}).get("stage0_busy_age_ms")
            if (
                self.joint_stage0_busy_hold_ms > 0.0
                and isinstance(stage0_busy_age_ms, (int, float))
                and float(stage0_busy_age_ms) <= self.joint_stage0_busy_hold_ms
            ):
                out.update({"action": "defer", "reason": "monitor_stage0_busy_hold"})
                return out

        deadline = self._deadline_decision(lead_ms, deadline_ctx)
        out["deadline"] = deadline
        if not deadline["admitted"]:
            out.update({"action": "reject", "reason": "deadline_miss"})
            return out
        return out

    def _deadline_decision(
        self,
        lead_ms: float | None,
        deadline_ctx: dict[str, Any],
    ) -> dict[str, Any]:
        estimate_ms = float(deadline_ctx["estimate_ms"])
        budget_ms = None if lead_ms is None else float(lead_ms) - self.deadline_guard_ms
        admitted = bool(budget_ms is not None and estimate_ms <= budget_ms)
        return {
            "admitted": admitted,
            "lead_ms": lead_ms,
            "guard_ms": self.deadline_guard_ms,
            "budget_ms": budget_ms,
            "estimate_ms": estimate_ms,
            "base_estimate_ms": deadline_ctx.get("base_estimate_ms"),
            "bucket_key": deadline_ctx.get("key"),
            "prefix_bucket": deadline_ctx.get("prefix_bucket"),
            "estimate_source": deadline_ctx.get("source"),
            "source_samples": deadline_ctx.get("source_samples"),
            "ewma_ms": self._prefetch_wall_ewma_ms,
            "samples": self._prefetch_wall_samples,
            "active_inflight": self._inflight,
            "inflight_penalty_ms": self.deadline_inflight_penalty_ms,
            "stage0_penalty_ms": self.deadline_stage0_penalty_ms,
        }

    async def register_active(
        self,
        *,
        request_id: str,
        engine_request_id: str | None,
        session_id: str | None,
        prefix_hash: str | None,
        admission: dict[str, Any] | None = None,
        lifecycle: Any = None,
    ) -> dict[str, Any]:
        deadline_ctx = (admission or {}).get("deadline_bucket")
        if not isinstance(deadline_ctx, dict):
            deadline_ctx = {}
        async with self._lock:
            self._active[request_id] = {
                "request_id": request_id,
                "engine_request_id": engine_request_id,
                "lifecycle": lifecycle,
                "session_id": session_id,
                "committed_prefix_hash": prefix_hash,
                "started_monotonic": time.monotonic(),
                "abort_requested": False,
                "admission_reason": (admission or {}).get("reason"),
                "deadline_bucket_key": deadline_ctx.get("key"),
                "deadline_prefix_bucket_key": deadline_ctx.get("prefix_bucket_key"),
                "deadline_prefix_bucket": deadline_ctx.get("prefix_bucket"),
            }
            return self._controller_stats()

    async def finish(
        self,
        request_id: str | None = None,
        *,
        terminal_status: str = "finished",
        observed_wall_s: float | None = None,
        release_slot: bool = False,
    ) -> dict[str, Any]:
        async with self._lock:
            # Active bookkeeping also tracks requests that bypass admission.
            # Only the lifecycle owning a successful reservation releases it.
            if release_slot:
                self._inflight = max(0, self._inflight - 1)
            if (
                terminal_status == "success"
                and isinstance(observed_wall_s, (int, float))
                and observed_wall_s > 0
            ):
                observed_ms = float(observed_wall_s) * 1000.0
                if self._prefetch_wall_samples <= 0:
                    self._prefetch_wall_ewma_ms = observed_ms
                else:
                    self._prefetch_wall_ewma_ms = (
                        self.deadline_alpha * observed_ms
                        + (1.0 - self.deadline_alpha)
                        * self._prefetch_wall_ewma_ms
                    )
                self._prefetch_wall_samples += 1
            if request_id is not None:
                entry = self._active.pop(request_id, None)
                if entry is not None:
                    entry["terminal_status"] = terminal_status
                    if (
                        terminal_status == "success"
                        and isinstance(observed_wall_s, (int, float))
                        and observed_wall_s > 0
                    ):
                        observed_ms = float(observed_wall_s) * 1000.0
                        self._update_deadline_bucket(
                            entry.get("deadline_bucket_key"),
                            observed_ms,
                        )
                        self._update_deadline_bucket(
                            entry.get("deadline_prefix_bucket_key"),
                            observed_ms,
                        )
            return self._controller_stats()

    def _update_deadline_bucket(self, key: Any, observed_ms: float) -> None:
        if not isinstance(key, str) or not key:
            return
        stat = self._deadline_bucket_stats.get(key)
        if stat is None:
            self._deadline_bucket_stats[key] = {
                "ewma_ms": float(observed_ms),
                "samples": 1,
            }
            return
        stat["ewma_ms"] = (
            self.deadline_alpha * float(observed_ms)
            + (1.0 - self.deadline_alpha) * float(stat.get("ewma_ms", observed_ms))
        )
        stat["samples"] = int(stat.get("samples", 0) or 0) + 1

    async def is_abort_requested(self, request_id: str) -> bool:
        async with self._lock:
            entry = self._active.get(request_id)
            return bool(entry and entry.get("abort_requested"))

    def _matches_live(
        self,
        entry: dict[str, Any],
        *,
        session_id: str | None,
        prefix_hash: str | None,
    ) -> bool:
        if self.abort_scope == "all":
            return True
        if self.abort_scope == "session":
            return bool(session_id and entry.get("session_id") == session_id)
        return bool(
            session_id
            and prefix_hash
            and entry.get("session_id") == session_id
            and entry.get("committed_prefix_hash") == prefix_hash
        )

    async def abort_for_live(
        self,
        state: State,
        *,
        session_id: str | None,
        prefix_hash: str | None,
    ) -> dict[str, Any]:
        decision: dict[str, Any] = {
            "enabled": self.abort_on_live,
            "scope": self.abort_scope,
            "aborted": 0,
            "request_ids": [],
            "engine_request_ids": [],
        }
        if not self.abort_on_live:
            return decision

        lifecycles = []
        async with self._lock:
            entries = [
                entry
                for entry in self._active.values()
                if not entry.get("abort_requested")
                and self._matches_live(
                    entry,
                    session_id=session_id,
                    prefix_hash=prefix_hash,
                )
            ]
            for entry in entries:
                entry["abort_requested"] = True
                lifecycle = entry.get("lifecycle")
                if lifecycle is not None:
                    # serving_chat determines the actual ID (including header
                    # overrides). Before metadata exists, cancel the owning
                    # handler so an early engine abort cannot miss submission.
                    entry["engine_request_id"] = lifecycle.owned_engine_request_id()
                    lifecycles.append(lifecycle)
            abort_ids = [str(entry["engine_request_id"]) for entry in entries if entry.get("engine_request_id")]
            self._live_abort_calls += 1
            self._live_abort_requests += len(entries)
            decision.update(
                {
                    "aborted": len(entries),
                    "request_ids": [entry["request_id"] for entry in entries],
                    "engine_request_ids": abort_ids,
                    "controller": self._controller_stats(),
                }
            )

        for lifecycle in lifecycles:
            lifecycle.request_live_abort()
        if not abort_ids:
            return decision

        engine_client = getattr(state, "engine_client", None)
        try:
            await engine_client.abort(abort_ids)
        except Exception:
            logger.exception(
                "OMNI prefetch live-abort failed for request_ids=%s",
                abort_ids,
            )
            decision["error"] = "abort_failed"
        return decision

    def resolve_priority(
        self,
        request: OmniPrefetchSessionRequest,
        state: State,
    ) -> tuple[int, str]:
        priority = self.default_priority if request.prefetch_priority is None else int(request.prefetch_priority)
        if priority == 0:
            return 0, "priority_disabled"
        policy = _scheduler_policy_name(state)
        if "priority" not in policy:
            logger.warning(
                "OMNI_PREFETCH_PRIORITY=%d requested but scheduler_policy=%r; using priority=0",
                priority,
                policy,
            )
            return 0, "scheduler_policy_not_priority"
        return priority, "priority_scheduler"

    def _deadline_stats_snapshot(self) -> dict[str, Any]:
        return {
            key: {
                "ewma_ms": round(float(value.get("ewma_ms", 0.0)), 1),
                "samples": int(value.get("samples", 0) or 0),
            }
            for key, value in sorted(self._deadline_bucket_stats.items())
        }

    def _controller_stats(self) -> dict[str, Any]:
        return {
            "inflight": self._inflight,
            "accepted": self._accepted,
            "skipped": self._skipped,
            "active": len(self._active),
            "joint_admitted": self._joint_admitted,
            "joint_skipped": self._joint_skipped,
            "joint_waits": self._joint_waits,
            "joint_wait_admitted": self._joint_wait_admitted,
            "joint_wait_timeout": self._joint_wait_timeout,
            "live_abort_calls": self._live_abort_calls,
            "live_abort_requests": self._live_abort_requests,
            "prefetch_wall_ewma_ms": round(self._prefetch_wall_ewma_ms),
            "prefetch_wall_samples": self._prefetch_wall_samples,
            "deadline_bucket_stats": self._deadline_stats_snapshot(),
        }


def _get_prefetch_admission_controller(state: State) -> OmniPrefetchAdmissionController:
    controller = getattr(state, "omni_prefetch_admission", None)
    if controller is None:
        controller = OmniPrefetchAdmissionController()
        state.omni_prefetch_admission = controller
    return controller


class OmniPrefetchRegistry:
    """API completion candidates; strict mode does not assert KV residency."""

    def __init__(self) -> None:
        self.route_hint_only = _env_bool("OMNI_PREFETCH_REPLICA_AFFINITY", False)
        self.enabled = _env_bool("OMNI_PREFETCH_FAST_BYPASS", False) or self.route_hint_only
        self.skip_offload_on_miss = _env_bool("OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS", False)
        self.history_hit_enable = _env_bool("OMNI_PREFETCH_HISTORY_HIT_ENABLE", False)
        self.ttl_s = max(1.0, _env_float("OMNI_PREFETCH_REGISTRY_TTL_SEC", 300.0))
        self._warm: dict[tuple[str, str], dict[str, Any]] = {}
        self._live_hits = 0
        self._live_misses = 0
        self._live_bypasses = 0
        self._history_only_ignored = 0
        self._fallback_hits = 0
        self._alias_marks = 0
        self._live_route_candidates = 0

    @staticmethod
    def _key(session_id: str | None, prefix_hash: str | None) -> tuple[str, str] | None:
        if not session_id or not prefix_hash:
            return None
        return (str(session_id), str(prefix_hash))

    def mark_warm(
        self,
        *,
        session_id: str | None,
        prefix_hash: str | None,
        alias_prefix_hashes: list[str] | None = None,
        request_id: str,
        admission: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        primary_key = self._key(session_id, prefix_hash)
        now = time.monotonic()
        scope = str((metadata or {}).get("prefetch_scope") or "")
        include_current_user = bool(
            isinstance(metadata, dict)
            and (
                metadata.get("prefetch_include_current_user")
                or scope in {"current_turn_prefill", "full_prompt"}
            )
        )
        gpu_pin_requested = bool(
            isinstance(metadata, dict) and metadata.get("gpu_pin_requested")
        )
        gpu_pin_ttl_ms = (
            metadata.get("gpu_pin_ttl_ms") if isinstance(metadata, dict) else None
        )
        stored = 0
        keys: list[tuple[tuple[str, str], str]] = []
        if primary_key is not None:
            keys.append((primary_key, "primary"))
        for alias_hash in alias_prefix_hashes or []:
            alias_key = self._key(session_id, alias_hash)
            if alias_key is None or any(existing == alias_key for existing, _ in keys):
                continue
            keys.append((alias_key, "alias"))
        for key, key_kind in keys:
            self._warm[key] = {
                "request_id": request_id,
                "created_monotonic": now,
                "expires_monotonic": now + self.ttl_s,
                "admission_reason": (admission or {}).get("reason"),
                "include_current_user": include_current_user,
                "prefetch_scope": scope or None,
                "key_kind": key_kind,
                "primary_prefix_hash": prefix_hash,
                "gpu_pin_requested": gpu_pin_requested,
                "gpu_pin_ttl_ms": gpu_pin_ttl_ms,
            }
            stored += 1
            if key_kind == "alias":
                self._alias_marks += 1
        return self.stats(
            extra={
                "marked": primary_key is not None,
                "marked_keys": stored,
                "alias_prefix_hashes": len(alias_prefix_hashes or []),
            }
        )

    def resolve_live(
        self,
        *,
        session_id: str | None,
        prefix_hash: str | None,
        fallback_prefix_hashes: list[str] | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic()
        decision: dict[str, Any] = {
            "enabled": self.enabled,
            "session_id": session_id,
            "committed_prefix_hash": prefix_hash,
            "matched_prefix_hash": None,
            "lookup_key_kind": None,
            "warm_scope": None,
            "gpu_pin_requested": False,
            "prefetch_hit_used": False,
            "skip_offload": False,
            "lookup_state": "disabled",
            "route_hint_only": self.route_hint_only,
            "cache_reuse_confirmed": None,
            "cache_residency_confirmed": None,
            "candidate_state": None,
        }
        if not self.enabled:
            return decision

        candidates: list[tuple[str, str]] = []
        if prefix_hash:
            candidates.append(("primary", str(prefix_hash)))
        for fallback_hash in fallback_prefix_hashes or []:
            if not fallback_hash:
                continue
            fallback = str(fallback_hash)
            if fallback not in {h for _, h in candidates}:
                candidates.append(("fallback", fallback))

        if not candidates:
            self._live_bypasses += 1
            decision.update(
                {
                    "lookup_state": "no_key",
                    "skip_offload": self.skip_offload_on_miss and not self.route_hint_only,
                }
            )
            return decision

        last_state = "cold"
        last_entry: dict[str, Any] | None = None
        for candidate_kind, candidate_hash in candidates:
            key = self._key(session_id, candidate_hash)
            if key is None:
                continue
            entry = self._warm.get(key)
            if entry is None:
                continue
            last_entry = entry
            if float(entry.get("expires_monotonic", 0.0)) < now:
                self._warm.pop(key, None)
                last_state = "expired"
                continue
            if not bool(entry.get("include_current_user")) and not self.history_hit_enable and not self.route_hint_only:
                self._history_only_ignored += 1
                last_state = "warm_history_only_ignored"
                continue

            if self.route_hint_only:
                self._live_route_candidates += 1
            else:
                self._live_hits += 1
            if candidate_kind == "fallback" and not self.route_hint_only:
                self._fallback_hits += 1
            decision.update(
                {
                    "lookup_state": (
                        "warm_route_candidate" if self.route_hint_only
                        else ("warm" if candidate_kind == "primary" else "warm_fallback")
                    ),
                    "prefetch_hit_used": not self.route_hint_only,
                    "skip_offload": False,
                    "warm_request_id": entry.get("request_id"),
                    "matched_prefix_hash": candidate_hash,
                    "lookup_key_kind": candidate_kind,
                    "warm_scope": entry.get("prefetch_scope"),
                    "warm_key_kind": entry.get("key_kind"),
                    "gpu_pin_requested": bool(entry.get("gpu_pin_requested")),
                    "candidate_state": "api_completed_candidate" if self.route_hint_only else None,
                }
            )
            return decision

        self._live_misses += 1
        self._live_bypasses += 1
        decision.update(
            {
                "lookup_state": last_state,
                "skip_offload": self.skip_offload_on_miss and not self.route_hint_only,
            }
        )
        if last_entry is not None:
            decision["warm_request_id"] = last_entry.get("request_id")
            decision["warm_scope"] = last_entry.get("prefetch_scope")
            decision["gpu_pin_requested"] = bool(
                last_entry.get("gpu_pin_requested")
            )
        return decision

    def stats(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        now = time.monotonic()
        expired = [key for key, entry in self._warm.items() if float(entry.get("expires_monotonic", 0.0)) < now]
        for key in expired:
            self._warm.pop(key, None)
        out = {
            "enabled": self.enabled,
            "skip_offload_on_miss": self.skip_offload_on_miss,
            "ttl_s": self.ttl_s,
            "warm_entries": len(self._warm),
            "live_hits": self._live_hits,
            "live_misses": self._live_misses,
            "live_bypasses": self._live_bypasses,
            "history_hit_enable": self.history_hit_enable,
            "history_only_ignored": self._history_only_ignored,
            "fallback_hits": self._fallback_hits,
            "alias_marks": self._alias_marks,
            "route_hint_only": self.route_hint_only,
            "live_route_candidates": self._live_route_candidates,
            "expired_pruned": len(expired),
        }
        if extra:
            out.update(extra)
        return out


def _get_prefetch_registry(state: State) -> OmniPrefetchRegistry:
    registry = getattr(state, "omni_prefetch_registry", None)
    if registry is None:
        registry = OmniPrefetchRegistry()
        state.omni_prefetch_registry = registry
    return registry


async def _consume_prefetch_stream(generator: AsyncIterator[str]) -> dict[str, Any]:
    chunks = 0
    text_len = 0
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    first_chunk_s: float | None = None
    sse_done = False
    terminal_finish_reasons: list[str] = []
    stream_errors: list[str] = []
    t0 = time.perf_counter()

    async for raw_chunk in generator:
        now = time.perf_counter()
        if first_chunk_s is None:
            first_chunk_s = now - t0
        chunks += 1
        for line in str(raw_chunk).splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            if data == "[DONE]":
                sse_done = True
                continue
            if sse_done:
                stream_errors.append("data_after_DONE")
            try:
                payload = json.loads(data)
            except Exception:
                stream_errors.append("invalid_JSON")
                continue
            if not isinstance(payload, dict):
                stream_errors.append("non_object_payload")
                continue
            if payload.get("error"):
                stream_errors.append("error_payload")
            if payload.get("usage") is not None:
                usage = payload.get("usage")
            for choice in payload.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    terminal_finish_reasons.append(choice["finish_reason"])
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    text_len += len(content)

    # Python 3.11+ exposes cancellation even when serving_chat swallowed it.
    # Python 3.10 lacks that public counter; propagated cancellation is still
    # cleaned up, but a cancellation already swallowed downstream is unknown.
    cancelling = getattr(asyncio.current_task(), "cancelling", None)
    if callable(cancelling) and cancelling():
        raise asyncio.CancelledError

    return {
        "chunks": chunks,
        "text_len": text_len,
        "usage": usage,
        "finish_reason": finish_reason,
        "sse_done": sse_done,
        "terminal_finish_reasons": terminal_finish_reasons,
        "stream_errors": stream_errors,
        "completion_confirmed": sse_done and bool(terminal_finish_reasons)
        and all(reason in ("stop", "length") for reason in terminal_finish_reasons) and not stream_errors,
        "first_chunk_s": first_chunk_s,
        "wall_s": time.perf_counter() - t0,
    }


async def _abort_prefetch_engine_request(state: State, engine_request_id: str) -> None:
    engine_client = getattr(state, "engine_client", None)
    if engine_client is None:
        return
    try:
        await engine_client.abort([engine_request_id])
    except Exception:
        logger.exception(
            "OMNI prefetch abort failed for engine_request_id=%s",
            engine_request_id,
        )


class _PrefetchRequestLifecycle:
    """Own one admission slot and its request-local generation cleanup."""

    def __init__(
        self, controller: Any, request_id: str, raw_request: Request, *, reserved_slot: bool = False,
    ) -> None:
        self.controller = controller
        self.request_id = request_id
        self.raw_request = raw_request
        self.generator: Any = None
        self.reserved_slot = reserved_slot
        self._owner_task = asyncio.current_task()
        self._initial_metadata = getattr(getattr(raw_request, "state", None), "request_metadata", None)
        self._finish_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    async def finish(self, terminal_status: str, observed_wall_s: float | None = None) -> dict[str, Any]:
        # A cancellation while finish waits for the controller lock must neither
        # lose the decrement nor make error cleanup decrement a second request.
        if self._finish_task is None:
            self._finish_task = asyncio.create_task(self.controller.finish(
                self.request_id, terminal_status=terminal_status, observed_wall_s=observed_wall_s,
                release_slot=self.reserved_slot))
        return await asyncio.shield(self._finish_task)

    def owned_engine_request_id(self) -> str | None:
        metadata = getattr(getattr(self.raw_request, "state", None), "request_metadata", None)
        engine_request_id = getattr(metadata, "request_id", None)
        if metadata is not self._initial_metadata and isinstance(engine_request_id, str) and engine_request_id:
            return engine_request_id
        return None

    def request_live_abort(self) -> None:
        # Opt-in live preemption must also stop preprocessing/first submission,
        # where an engine-only abort can arrive before the request exists. This
        # may cancel the HTTP response instead of returning an ABORTED JSON body.
        task = self._owner_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel("prefetch superseded by live request")

    async def _cleanup(self, terminal_status: str) -> None:
        try:
            engine_request_id = self.owned_engine_request_id()
            # serving_chat sets fresh metadata before submitting generation.
            # Do not guess an ID from the payload: X-Request-Id can override it.
            if engine_request_id is not None:
                try:
                    await _abort_prefetch_engine_request(self.raw_request.app.state, engine_request_id)
                except (Exception, asyncio.CancelledError):
                    logger.exception("OMNI prefetch request cleanup abort failed: %s", self.request_id)
            close = getattr(self.generator, "aclose", None)
            if callable(close):
                try:
                    await close()
                except (Exception, asyncio.CancelledError):
                    logger.exception("OMNI prefetch generator cleanup failed: %s", self.request_id)
        finally:
            await self.finish(terminal_status)

    async def cleanup(self, terminal_status: str) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup(terminal_status))
        # Keep a strong reference and await the owned task even if the HTTP
        # wrapper cancels us again. The caller re-raises its original exception.
        while not self._cleanup_task.done():
            try:
                await asyncio.shield(self._cleanup_task)
            except asyncio.CancelledError:
                continue
        self._cleanup_task.result()


@router.post(
    "/v1/omni/prefetch_session",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def omni_prefetch_session(request: OmniPrefetchSessionRequest, raw_request: Request):
    if trace_enabled():
        emit("server_accept", request_id=request.request_id,
             client_request_id=raw_request.headers.get("x-request-id"), request_kind="prefetch")
    handler = Omnichat(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Chat Completions API",
            )
        return base_server.create_error_response(message="The model does not support Chat Completions API")

    model = request.model or getattr(getattr(raw_request.app.state, "args", None), "model", None)
    if not model:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="model is required")

    prefetch_request_id = request.request_id or f"omni-prefetch-{random_uuid()}"
    controller = _get_prefetch_admission_controller(raw_request.app.state)
    admitted, admission = await controller.decide(request, raw_request.app.state)
    lifecycle = _PrefetchRequestLifecycle(
        controller, prefetch_request_id, raw_request,
        reserved_slot=admitted and bool(admission.get("inflight_reserved", False)),
    )
    try:
        return await _execute_prefetch_session(request, raw_request, handler, model, admitted, admission, lifecycle)
    except (Exception, asyncio.CancelledError) as exc:
        if admitted:
            try:
                await lifecycle.cleanup("cancelled" if isinstance(exc, asyncio.CancelledError) else "error")
            except (Exception, asyncio.CancelledError):
                logger.exception("OMNI prefetch admission cleanup failed: %s", prefetch_request_id)
        raise


async def _execute_prefetch_session(
    request: OmniPrefetchSessionRequest,
    raw_request: Request,
    handler: Any,
    model: str,
    admitted: bool,
    admission: dict[str, Any],
    lifecycle: _PrefetchRequestLifecycle,
):
    controller = lifecycle.controller
    prefetch_request_id = lifecycle.request_id
    registry = _get_prefetch_registry(raw_request.app.state)
    monitor = _get_runtime_monitor(raw_request.app.state)
    priority, priority_reason = controller.resolve_priority(request, raw_request.app.state)
    admission["priority"] = priority
    admission["priority_reason"] = priority_reason
    await monitor.record_prefetch_decision(
        request_id=prefetch_request_id,
        session_id=request.session_id,
        prefix_hash=request.committed_prefix_hash,
        admitted=admitted,
        admission=admission,
    )
    if not admitted:
        return JSONResponse(
            content={
                "status": "SKIPPED",
                "strategy": "admission_control",
                "request_id": prefetch_request_id,
                "session_id": request.session_id,
                "committed_prefix_hash": request.committed_prefix_hash,
                "prefix_token_estimate": request.prefix_token_estimate,
                "lead_ms": request.lead_ms,
                "admission": admission,
                "registry": registry.stats(),
                "skip_reason": admission.get("reason"),
                "wall_s": 0.0,
            }
        )

    prefetch_metadata = dict(request.metadata or {})
    prefetch_eviction_metadata = {
        field_name: prefetch_metadata[field_name]
        for field_name in _OMNI_EVICTION_METADATA_FIELDS
        if prefetch_metadata.get(field_name) is not None
    }
    prefetch_eviction_metadata.update(
        {
            "omni_prefetch_request_id": prefetch_request_id,
            "omni_prefetch_session_id": request.session_id,
            "omni_prefetch_committed_prefix_hash": request.committed_prefix_hash,
            "omni_prefetch_prefix_token_estimate": request.prefix_token_estimate,
            "omni_prefetch_lead_ms": request.lead_ms,
            "omni_prefetch_prefetch_scope": prefetch_metadata.get("prefetch_scope"),
        }
    )
    gpu_pin_requested = _env_bool("OMNI_PREFETCH_GPU_PIN", False)
    gpu_pin_ttl_ms = _env_float("OMNI_PREFETCH_GPU_PIN_TTL_MS", 5000.0)
    if gpu_pin_requested:
        prefetch_metadata["gpu_pin_requested"] = True
        prefetch_metadata["gpu_pin_ttl_ms"] = gpu_pin_ttl_ms

    payload = {
        "model": model,
        "messages": request.messages,
        "stream": True,
        "stream_options": request.stream_options or {"include_usage": True},
        "request_id": prefetch_request_id,
        "modalities": ["text"],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "priority": priority,
    }
    payload.update(prefetch_eviction_metadata)

    t0 = time.perf_counter()
    await controller.register_active(
        request_id=prefetch_request_id,
        engine_request_id=None,
        session_id=request.session_id,
        prefix_hash=request.committed_prefix_hash,
        admission=admission,
        lifecycle=lifecycle,
    )
    await monitor.record_prefetch_start(
        request_id=prefetch_request_id,
        session_id=request.session_id,
        prefix_hash=request.committed_prefix_hash,
        admission=admission,
    )

    async def finish_prefetch(
        terminal_status: str,
        observed_wall_s: float | None = None,
    ) -> dict[str, Any]:
        controller_stats = await lifecycle.finish(terminal_status, observed_wall_s)
        await monitor.record_prefetch_finish(
            request_id=prefetch_request_id,
            session_id=request.session_id,
            prefix_hash=request.committed_prefix_hash,
            terminal_status=terminal_status,
            wall_s=observed_wall_s,
            controller=controller_stats,
        )
        return controller_stats

    try:
        chat_request = ChatCompletionRequest.model_validate(payload)
        prefetch_kv_policy = {
            "enabled": False,
            "eviction_metadata": prefetch_eviction_metadata,
        }
        scheduler_prefetch_policy = {
            "enabled": _env_bool("OMNI_SCHED_NATIVE_PREFETCH", False),
            "request_id": prefetch_request_id,
            "session_id": request.session_id,
            "committed_prefix_hash": request.committed_prefix_hash,
            "prefix_token_estimate": request.prefix_token_estimate,
            "lead_ms": request.lead_ms,
            "priority": priority,
            "cancellable": True,
            "gpu_pin": gpu_pin_requested,
            "gpu_pin_ttl_ms": gpu_pin_ttl_ms,
            "prefetch_scope": prefetch_metadata.get("prefetch_scope"),
        }
        try:
            setattr(chat_request, "omni_kv_policy", prefetch_kv_policy)
            setattr(
                chat_request,
                "omni_scheduler_prefetch_policy",
                scheduler_prefetch_policy,
            )
            setattr(chat_request, "priority", priority)
        except Exception:
            object.__setattr__(chat_request, "omni_kv_policy", prefetch_kv_policy)
            object.__setattr__(
                chat_request,
                "omni_scheduler_prefetch_policy",
                scheduler_prefetch_policy,
            )
            object.__setattr__(chat_request, "priority", priority)
        generator = await handler.create_chat_completion(chat_request, raw_request)
        lifecycle.generator = generator
        # Task.cancelling() is only available on Python 3.11+ (see stream path).
        cancelling = getattr(asyncio.current_task(), "cancelling", None)
        if callable(cancelling) and cancelling():
            raise asyncio.CancelledError
    except (EngineGenerateError, EngineDeadError) as exc:
        if await controller.is_abort_requested(prefetch_request_id):
            return JSONResponse(
                content={
                    "status": "ABORTED",
                    "strategy": "live_abort",
                    "request_id": prefetch_request_id,
                    "session_id": request.session_id,
                    "committed_prefix_hash": request.committed_prefix_hash,
                    "prefix_token_estimate": request.prefix_token_estimate,
                    "lead_ms": request.lead_ms,
                    "admission": admission,
                    "controller": await finish_prefetch("aborted"),
                    "registry": registry.stats(),
                    "skip_reason": "live_abort",
                    "wall_s": time.perf_counter() - t0,
                }
            )
        await finish_prefetch("engine_error")
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as exc:
        await finish_prefetch("error")
        logger.exception("Omni prefetch session failed before generation: %s", exc)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(exc)) from exc

    if isinstance(generator, ErrorResponse):
        await finish_prefetch("error_response")
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.error.code if generator.error else 400,
        )

    if isinstance(generator, ChatCompletionResponse):
        if await controller.is_abort_requested(prefetch_request_id):
            return JSONResponse(
                content={
                    "status": "ABORTED",
                    "strategy": "live_abort",
                    "request_id": prefetch_request_id,
                    "session_id": request.session_id,
                    "committed_prefix_hash": request.committed_prefix_hash,
                    "prefix_token_estimate": request.prefix_token_estimate,
                    "lead_ms": request.lead_ms,
                    "wall_s": time.perf_counter() - t0,
                    "non_stream_response": True,
                    "admission": admission,
                    "controller": await finish_prefetch("aborted"),
                    "registry": registry.stats(),
                    "skip_reason": "live_abort",
                }
            )
        if registry.route_hint_only and not (
            generator.choices and all(choice.finish_reason in ("stop", "length") for choice in generator.choices)
        ):
            return JSONResponse(content={
                "status": "FAILED", "strategy": "warm_request", "request_id": prefetch_request_id,
                "skip_reason": "prefetch_completion_unconfirmed", "completion_confirmed": False,
                "admission": admission, "controller": await finish_prefetch("completion_unconfirmed"),
                "registry": registry.stats(), "non_stream_response": True,
            })
        registry_stats = registry.mark_warm(
            session_id=request.session_id,
            prefix_hash=request.committed_prefix_hash,
            alias_prefix_hashes=request.alias_prefix_hashes,
            request_id=prefetch_request_id,
            admission=admission,
            metadata=prefetch_metadata,
        )
        wall_s = time.perf_counter() - t0
        return JSONResponse(
            content={
                "status": "SUCCESS",
                "strategy": "warm_request",
                "request_id": prefetch_request_id,
                "session_id": request.session_id,
                "committed_prefix_hash": request.committed_prefix_hash,
                "prefix_token_estimate": request.prefix_token_estimate,
                "lead_ms": request.lead_ms,
                "wall_s": wall_s,
                "usage": generator.usage.model_dump(mode="json") if generator.usage is not None else None,
                "non_stream_response": True,
                "admission": admission,
                "controller": await finish_prefetch("success", wall_s),
                "registry": registry_stats,
            }
        )

    try:
        stream_timeout_ms = _env_float("OMNI_PREFETCH_STREAM_TIMEOUT_MS", 0.0)
        stream_task = _consume_prefetch_stream(cast(AsyncIterator[str], generator))
        if stream_timeout_ms > 0.0:
            stream_stats = await asyncio.wait_for(
                stream_task,
                timeout=stream_timeout_ms / 1000.0,
            )
        else:
            stream_stats = await stream_task
    except asyncio.TimeoutError:
        await lifecycle.cleanup("scheduler_timeout")
        wall_s = time.perf_counter() - t0
        return JSONResponse(
            content={
                "status": "ABORTED",
                "strategy": "scheduler_native_cancel",
                "request_id": prefetch_request_id,
                "session_id": request.session_id,
                "committed_prefix_hash": request.committed_prefix_hash,
                "prefix_token_estimate": request.prefix_token_estimate,
                "lead_ms": request.lead_ms,
                "admission": admission,
                "controller": await finish_prefetch("scheduler_timeout", wall_s),
                "registry": registry.stats(),
                "skip_reason": "prefetch_stream_timeout",
                "stream_timeout_ms": stream_timeout_ms,
                "wall_s": wall_s,
            }
        )
    except (EngineGenerateError, EngineDeadError) as exc:
        if await controller.is_abort_requested(prefetch_request_id):
            return JSONResponse(
                content={
                    "status": "ABORTED",
                    "strategy": "live_abort",
                    "request_id": prefetch_request_id,
                    "session_id": request.session_id,
                    "committed_prefix_hash": request.committed_prefix_hash,
                    "prefix_token_estimate": request.prefix_token_estimate,
                    "lead_ms": request.lead_ms,
                    "admission": admission,
                    "controller": await finish_prefetch("aborted"),
                    "registry": registry.stats(),
                    "skip_reason": "live_abort",
                    "wall_s": time.perf_counter() - t0,
                }
            )
        await finish_prefetch("engine_error")
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as exc:
        await finish_prefetch("error")
        logger.exception("Omni prefetch session stream failed: %s", exc)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(exc)) from exc

    if await controller.is_abort_requested(prefetch_request_id):
        return JSONResponse(
            content={
                "status": "ABORTED",
                "strategy": "live_abort",
                "request_id": prefetch_request_id,
                "session_id": request.session_id,
                "committed_prefix_hash": request.committed_prefix_hash,
                "prefix_token_estimate": request.prefix_token_estimate,
                "lead_ms": request.lead_ms,
                "admission": admission,
                "controller": await finish_prefetch("aborted"),
                "registry": registry.stats(),
                "skip_reason": "live_abort",
                **stream_stats,
            }
        )

    if registry.route_hint_only and not stream_stats["completion_confirmed"]:
        return JSONResponse(content={
            "status": "FAILED", "strategy": "warm_request", "request_id": prefetch_request_id,
            "skip_reason": "prefetch_completion_unconfirmed", "admission": admission,
            "controller": await finish_prefetch("completion_unconfirmed"), "registry": registry.stats(),
            **stream_stats,
        })
    registry_stats = registry.mark_warm(
        session_id=request.session_id,
        prefix_hash=request.committed_prefix_hash,
        alias_prefix_hashes=request.alias_prefix_hashes,
        request_id=prefetch_request_id,
        admission=admission,
        metadata=prefetch_metadata,
    )
    wall_s = time.perf_counter() - t0
    stream_stats["wall_s"] = wall_s
    return JSONResponse(
        content={
            "status": "SUCCESS",
            "strategy": "warm_request",
            "request_id": prefetch_request_id,
            "session_id": request.session_id,
            "committed_prefix_hash": request.committed_prefix_hash,
            "prefix_token_estimate": request.prefix_token_estimate,
            "lead_ms": request.lead_ms,
            "admission": admission,
            "controller": await finish_prefetch("success", wall_s),
            "registry": registry_stats,
            **stream_stats,
        }
    )


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    if trace_enabled():
        emit("server_accept", request_id=getattr(request, "request_id", None),
             client_request_id=raw_request.headers.get("x-request-id"), request_kind="live")
    metrics_header_format = raw_request.headers.get(ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, "")
    handler = Omnichat(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Chat Completions API",
            )
        return base_server.create_error_response(message="The model does not support Chat Completions API")
    raw_shadow_payload: dict[str, Any] = {}
    if (
        _env_bool("OMNI_PREFETCH_REPLICA_AFFINITY", False)
        or _env_bool("OMNI_KV_EVICT_SHADOW", False)
        or _env_policy_enabled("OMNI_KV_EVICT_ENFORCE_POLICY")
    ):
        try:
            raw_json = await raw_request.json()
        except Exception:
            raw_json = None
        if isinstance(raw_json, dict):
            raw_shadow_payload = raw_json

    def _request_or_shadow_payload(field_name: str) -> Any:
        value = getattr(request, field_name, None)
        if value is None and raw_shadow_payload:
            value = raw_shadow_payload.get(field_name)
        return value

    registry = _get_prefetch_registry(raw_request.app.state)
    controller = _get_prefetch_admission_controller(raw_request.app.state)
    monitor = _get_runtime_monitor(raw_request.app.state)
    session_id = _request_or_shadow_payload("omni_session_id")
    prefix_hash = _request_or_shadow_payload("committed_prefix_hash")
    fallback_prefix_hashes: list[str] = []
    history_prefix_hash = _request_or_shadow_payload("omni_history_prefix_hash")
    if isinstance(history_prefix_hash, str) and history_prefix_hash:
        fallback_prefix_hashes.append(history_prefix_hash)
    raw_fallbacks = _request_or_shadow_payload("omni_fallback_prefix_hashes")
    if isinstance(raw_fallbacks, str):
        fallback_prefix_hashes.extend(
            item.strip()
            for item in raw_fallbacks.replace(";", ",").split(",")
            if item.strip()
        )
    elif isinstance(raw_fallbacks, list):
        fallback_prefix_hashes.extend(
            str(item) for item in raw_fallbacks if item
        )
    kv_policy = registry.resolve_live(
        session_id=session_id,
        prefix_hash=prefix_hash,
        fallback_prefix_hashes=fallback_prefix_hashes,
    )
    prefix_token_estimate = _request_or_shadow_payload("prefix_token_estimate")
    if prefix_token_estimate is None:
        prefix_token_estimate = _request_or_shadow_payload(
            "omni_prefix_token_estimate"
        )
    history_prefix_token_estimate = _request_or_shadow_payload(
        "omni_history_prefix_token_estimate"
    )
    if (
        history_prefix_hash
        and kv_policy.get("matched_prefix_hash") == history_prefix_hash
        and history_prefix_token_estimate is not None
    ):
        prefix_token_estimate = history_prefix_token_estimate
    kv_policy["prefix_token_estimate"] = prefix_token_estimate
    kv_policy["fallback_prefix_hashes"] = fallback_prefix_hashes
    kv_policy["history_prefix_hash"] = history_prefix_hash
    kv_policy["history_prefix_token_estimate"] = history_prefix_token_estimate
    eviction_metadata: dict[str, Any] = {}
    for field_name in (
        "omni_round_index",
        "omni_trace_timestamp",
        "omni_query_length",
        "omni_response_length",
        "omni_history_tokens_before",
        "omni_history_tokens_after",
        "omni_prev_gap_sec",
        "omni_expected_next_gap_sec",
        "omni_expected_next_use_ts",
        "omni_audio_total_sec",
        "omni_audio_remaining_at_finish_sec",
        "omni_audio_reply_delay_sec",
        "omni_is_terminal_turn",
    ):
        value = _request_or_shadow_payload(field_name)
        if value is not None:
            eviction_metadata[field_name] = value
    if eviction_metadata:
        kv_policy["eviction_metadata"] = eviction_metadata
    kv_policy["live_abort"] = await controller.abort_for_live(
        raw_request.app.state,
        session_id=session_id,
        prefix_hash=prefix_hash,
    )
    if int((kv_policy.get("live_abort") or {}).get("aborted") or 0) > 0:
        await monitor.record_live_abort(
            session_id=session_id,
            prefix_hash=prefix_hash,
            abort=kv_policy["live_abort"],
        )
    kv_policy["runtime_monitor"] = await monitor.record_live_arrival(
        request_id=getattr(request, "request_id", None),
        session_id=session_id,
        prefix_hash=prefix_hash,
        prefix_token_estimate=prefix_token_estimate,
        kv_policy=kv_policy,
        snapshot=_build_omni_load_snapshot(raw_request.app.state),
    )
    setattr(request, "omni_kv_policy", kv_policy)
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        logger.exception("Chat completion failed: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.error.code if generator.error else 400,
        )

    elif isinstance(generator, ChatCompletionResponse):
        # Completely bypass Pydantic serialization warnings for multimodal content
        # by converting to dict first, then serializing with warnings suppressed
        import json as json_lib
        import warnings as warnings_module

        # Temporarily suppress ALL Pydantic UserWarnings during serialization
        with warnings_module.catch_warnings():
            warnings_module.filterwarnings("ignore", category=UserWarning)
            warnings_module.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
            try:
                # Use serialize_as_any=True to bypass type checking
                response_dict = generator.model_dump(mode="json", serialize_as_any=True, warnings="none")
                return JSONResponse(
                    content=response_dict,
                    headers=metrics_header(metrics_header_format),
                )
            except Exception:
                # Fallback: convert to JSON string and parse back to avoid any serialization issues
                try:
                    response_json = generator.model_dump_json(warnings="none", serialize_as_any=True)
                    response_dict = json_lib.loads(response_json)
                    return JSONResponse(
                        content=response_dict,
                        headers=metrics_header(metrics_header_format),
                    )
                except Exception:
                    # Last resort: regular dump with warnings suppressed
                    with warnings_module.catch_warnings():
                        warnings_module.filterwarnings("ignore", category=UserWarning)
                        return JSONResponse(
                            content=generator.model_dump(mode="json", warnings="none"),
                            headers=metrics_header(metrics_header_format),
                        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


_remove_route_from_router(router, "/v1/audio/speech", {"POST"})


@router.post(
    "/v1/audio/speech",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech(request: OpenAICreateSpeechRequest, raw_request: Request):
    """Generate speech audio from text using the loaded TTS model.

    Args:
        request: Speech synthesis request in OpenAI-compatible format.
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        The generated audio response, or an OpenAI-style error payload when
        the request cannot be fulfilled.

    Raises:
        HTTPException: If the server does not support speech generation or the
        synthesis request fails unexpectedly.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        return base_server.create_error_response(message="The model does not support Speech API")
    try:
        result = await handler.create_speech(request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code if result.error else 400,
            )
        return result
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.post(
    "/v1/audio/speech/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech_batch(request: BatchSpeechRequest, raw_request: Request):
    handler = Omnispeech(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        return base_server.create_error_response(message="The model does not support Speech API")
    try:
        result = await handler.create_speech_batch(request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code if result.error else 400,
            )
        return JSONResponse(content=result.model_dump())
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.post(
    "/v1/audio/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_audio_generate(request: OpenAICreateAudioGenerateRequest, raw_request: Request):
    handler = OmniAudioGenerate(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "openai_serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Audio Generate API",
            )
        return base_server.create_error_response(message="The model does not support Audio Generate API")
    try:
        result = await handler.create_audio_generate(request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code if result.error else 400,
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.get(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def list_voices(raw_request: Request):
    """List available TTS voices exposed by the loaded speech model.

    Args:
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        A JSON payload containing the sorted set of supported speaker names, or
        an OpenAI-style error response when the current server configuration
        does not support the Speech API.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Speech API")

    # Get all speakers (both model built-in and uploaded)
    speakers = sorted(handler.supported_speakers) if handler.supported_speakers else []

    # Get uploaded speakers details
    uploaded_speakers = []
    if hasattr(handler, "uploaded_speakers"):
        for voice_name, info in handler.uploaded_speakers.items():
            voice_entry = {
                "name": info.get("name", voice_name),
                "consent": info.get("consent", ""),
                "created_at": info.get("created_at", 0),
                "file_size": info.get("file_size", 0),
                "mime_type": info.get("mime_type", ""),
                "embedding_source": info.get("embedding_source", "audio"),
                "embedding_dim": info.get("embedding_dim"),
            }
            if info.get("ref_text"):
                voice_entry["ref_text"] = info["ref_text"]
            if info.get("speaker_description"):
                voice_entry["speaker_description"] = info["speaker_description"]
            uploaded_speakers.append(voice_entry)

    return JSONResponse(content={"voices": speakers, "uploaded_voices": uploaded_speakers})


@router.post(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def upload_voice(
    raw_request: Request,
    audio_sample: UploadFile | None = File(None),
    speaker_embedding: str | None = Form(None),
    consent: str = Form(...),
    name: str = Form(...),
    ref_text: str | None = Form(None),
    speaker_description: str | None = Form(None),
):
    """Upload a new voice for voice cloning.

    Accepts either an audio file or a pre-computed speaker embedding vector.
    These are mutually exclusive: provide one or the other.

    When using ``audio_sample``, the server stores the audio and extracts the
    speaker embedding on first use (Base task models only).

    When using ``speaker_embedding``, pass a JSON-encoded list of floats
    (1024-dim for 0.6B, 2048-dim for 1.7B). The voice is stored as a
    safetensors file and is immediately ready for use.

    Args:
        audio_sample: Audio file (max 10MB). Mutually exclusive with speaker_embedding.
        speaker_embedding: JSON-encoded float list. Mutually exclusive with audio_sample.
        consent: Consent recording ID
        name: Name for the new voice
        ref_text: Optional transcript of the audio for ICL (in-context
            learning) mode. When provided, voice clone requests using this
            voice will produce higher quality results.
        speaker_description: Optional free-form description of the voice
            (e.g. "warm speaker", "energetic narrator").
        raw_request: Raw FastAPI request

    Returns:
        JSON response with voice information
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Speech API")

    try:
        if speaker_embedding is not None and audio_sample is not None:
            return base(raw_request).create_error_response(
                message="'audio_sample' and 'speaker_embedding' are mutually exclusive"
            )
        if speaker_embedding is not None:
            result = await handler.upload_voice_embedding(speaker_embedding, consent, name)
        elif audio_sample is not None:
            result = await handler.upload_voice(
                audio_sample,
                consent,
                name,
                ref_text=ref_text,
                speaker_description=speaker_description,
            )
        else:
            return base(raw_request).create_error_response(
                message="Either 'audio_sample' or 'speaker_embedding' must be provided"
            )

        return JSONResponse(content={"success": True, "voice": result})

    except ValueError as e:
        return base(raw_request).create_error_response(message=str(e))
    except Exception as e:
        logger.exception(f"Failed to upload voice: {e}")
        return base(raw_request).create_error_response(message=f"Failed to upload voice: {str(e)}")


@router.delete(
    "/v1/audio/voices/{name}",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def delete_voice(name: str, raw_request: Request):
    """Delete an uploaded voice.

    Deletes the voice sample and associated metadata. Also removes any
    cached voice clone prompts for this voice.

    Args:
        name: Name of the voice to delete
        raw_request: Raw FastAPI request

    Returns:
        JSON response indicating success or failure
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(message="The model does not support Speech API")

    try:
        # Delete the voice
        success = await handler.delete_voice(name)
        if not success:
            return JSONResponse(
                content={"success": False, "error": f"Voice '{name}' not found"},
                status_code=HTTPStatus.NOT_FOUND.value,
            )

        return JSONResponse(content={"success": True, "message": f"Voice '{name}' deleted successfully"})

    except ValueError as e:
        return base(raw_request).create_error_response(message=str(e))
    except Exception as e:
        logger.exception(f"Failed to delete voice '{name}': {e}")
        return base(raw_request).create_error_response(message=f"Failed to delete voice: {str(e)}")


@router.websocket("/v1/audio/speech/stream")
async def streaming_speech(websocket: WebSocket):
    """WebSocket endpoint for streaming text input TTS.

    Accepts text incrementally, splits at sentence boundaries, and
    returns audio per sentence. See serving_speech_stream.py for protocol.
    """
    handler = getattr(websocket.app.state, "openai_streaming_speech", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming speech is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/video/chat/stream")
async def streaming_video_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming video input chat."""
    handler = getattr(websocket.app.state, "openai_streaming_video", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming video chat is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket):
    """WebSocket endpoint for OpenAI-style realtime interactions."""
    engine_client = getattr(websocket.app.state, "engine_client", None)
    if engine_client is not None and getattr(engine_client, "async_chunk", False):
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "error": (
                    "The /v1/realtime API is not supported when async_chunk is enabled on the server. "
                    "Use a stage configuration with async_chunk disabled and restart the server before using "
                    "this endpoint."
                ),
                "code": "unsupported",
            }
        )
        await websocket.close()
        return
    serving = getattr(websocket.app.state, "openai_serving_realtime", None)
    if serving is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Realtime API is not available", "code": "unsupported"})
        await websocket.close()
        return
    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()


# Health and Model endpoints for diffusion mode


# Remove existing health endpoint if present (from vllm imports)
# to ensure our handler takes precedence
_remove_route_from_router(router, "/health")


@router.get("/health")
async def health(raw_request: Request) -> JSONResponse:
    """Health check endpoint that works for both LLM and diffusion modes.

    Returns 200 OK if the server is healthy, 503 if the engine is dead.
    Mirrors vLLM upstream's /health which catches EngineDeadError -> 503.
    """
    engine_client = getattr(raw_request.app.state, "engine_client", None) or getattr(
        raw_request.app.state, "diffusion_engine", None
    )
    if engine_client is None:
        return JSONResponse(
            content={"status": "unhealthy", "reason": "No engine initialized"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )

    try:
        await engine_client.check_health()
        return JSONResponse(content={"status": "healthy"})
    except EngineDeadError:
        return JSONResponse(
            content={"status": "unhealthy"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )


# Remove existing models endpoint if present (from vllm imports)
# to ensure our handler takes precedence
_remove_route_from_router(router, "/v1/models")


@router.get("/v1/models")
async def show_available_models(raw_request: Request) -> JSONResponse:
    """Show available models for both LLM and diffusion modes.

    Delegates to state.openai_serving_models which is set to either
    OpenAIServingModels (LLM) or _DiffusionServingModels (pure diffusion).
    """
    handler = getattr(raw_request.app.state, "openai_serving_models", None)
    if handler is not None:
        models = await handler.show_available_models()
        return JSONResponse(content=models.model_dump())
    return JSONResponse(content={"object": "list", "data": []})


# Image generation API endpoints


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def generate_images(request: ImageGenerationRequest, raw_request: Request):
    """Generate images from text prompts using diffusion models.

    OpenAI DALL-E compatible endpoint for text-to-image generation.
    Only supports multi-stage omni mode with diffusion stages.

    Args:
        request: Image generation request with prompt and parameters
        raw_request: Raw FastAPI request for accessing app state

    Returns:
        ImageGenerationResponse with generated images as base64 PNG

    Raises:
        HTTPException: For validation errors, missing engine, or generation failures
    """
    # Get engine client (AsyncOmni) from app state
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)

    if request.model is not None and request.model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(f"Model mismatch: request specifies '{request.model}' but server is running '{model_name}'."),
        )

    try:
        # Unify request construction for any multi-stage pipeline to avoid
        # divergence between /v1/images and /v1/chat/completions.
        if len(stage_configs) > 1:
            chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_handler is None:
                logger.warning("openai_serving_chat is not initialized for multi-stage /v1/images/generations")
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="openai_serving_chat is not initialized for multi-stage image generation.",
                )

            effective_seed = request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
            extra_body: dict[str, Any] = {
                "seed": effective_seed,
                "num_outputs_per_prompt": request.n,
            }
            if request.size is not None:
                parse_size(request.size)
                width, height = parse_size(request.size)
                app_state_args = getattr(raw_request.app.state, "args", None)
                _check_max_generated_image_size(app_state_args, width, height)
                extra_body["size"] = request.size
            if request.negative_prompt is not None:
                extra_body["negative_prompt"] = request.negative_prompt
            if request.num_inference_steps is not None:
                extra_body["num_inference_steps"] = request.num_inference_steps
            if request.guidance_scale is not None:
                extra_body["guidance_scale"] = request.guidance_scale
            if request.true_cfg_scale is not None:
                extra_body["true_cfg_scale"] = request.true_cfg_scale
            if request.generator_device is not None:
                extra_body["generator_device"] = request.generator_device
            if request.lora is not None:
                # Keep /images validation semantics: invalid LoRA should fail with 400.
                _parse_lora_request(request.lora)
                extra_body["lora"] = request.lora

            generation_result = await chat_handler.generate_diffusion_images(
                prompt=request.prompt,
                extra_body=extra_body,
                request_id=f"img_gen-{random_uuid()}",
            )
            if isinstance(generation_result, ErrorResponse):
                return JSONResponse(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    content=generation_result.model_dump(),
                )
            flat_images, _, _ = generation_result
            image_data = [ImageData(b64_json=encode_image_base64(img), revised_prompt=None) for img in flat_images]
            return ImageGenerationResponse(created=int(time.time()), data=image_data)

        # Build params - pass through user values directly
        prompt: OmniTextPrompt = {"prompt": request.prompt}
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt
        gen_params = OmniDiffusionSamplingParams(num_outputs_per_prompt=request.n)
        extra_args = {}
        if request.use_system_prompt is not None:
            extra_args["use_system_prompt"] = request.use_system_prompt
        if request.system_prompt is not None:
            extra_args["system_prompt"] = request.system_prompt
        if extra_args:
            gen_params.extra_args = extra_args
        # Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        lora_request, lora_scale = _parse_lora_request(request.lora)
        _update_if_not_none(gen_params, "lora_request", lora_request)
        _update_if_not_none(gen_params, "lora_scale", lora_scale)

        # Parse and add size if provided
        width, height = None, None
        if request.size:
            width, height = parse_size(request.size)
            size_str = f"{width}x{height}"
        else:
            size_str = "model default"

        # Keep AR stage target grid in sync with requested output size.
        # GLM-Image consumes target_h/target_w via mm_processor_kwargs.
        if width is not None and height is not None:
            prompt["mm_processor_kwargs"] = {
                "target_h": height,
                "target_w": width,
            }
            # Backward-compatible fallback for processors reading top-level fields.
            prompt["height"] = height
            prompt["width"] = width
        app_state_args = getattr(raw_request.app.state, "args", None)
        _check_max_generated_image_size(app_state_args, width, height)

        _update_if_not_none(gen_params, "width", width)
        _update_if_not_none(gen_params, "height", height)

        # 3.3 Add optional parameters ONLY if provided
        _update_if_not_none(gen_params, "num_inference_steps", request.num_inference_steps)
        _update_if_not_none(gen_params, "guidance_scale", request.guidance_scale)
        _update_if_not_none(gen_params, "true_cfg_scale", request.true_cfg_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        _update_if_not_none(
            gen_params, "seed", request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
        )
        _update_if_not_none(gen_params, "generator_device", request.generator_device)
        _update_if_not_none(gen_params, "layers", request.layers)

        request_id = f"img_gen-{random_uuid()}"
        raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)

        logger.info(f"Generating {request.n} image(s) {size_str}")

        # Generate images using AsyncOmni (multi-stage mode)
        result = await _generate_with_async_omni(
            engine_client=engine_client,
            gen_params=gen_params,
            stage_configs=stage_configs,
            prompt=prompt,
            request_id=request_id,
        )

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from multi-stage pipeline.",
            )

        # Extract images from result
        images = _extract_images_from_result(result)

        logger.info(f"Successfully generated {len(images)} image(s)")

        # Determine output format (default to png)
        output_format = _choose_output_format(request.output_format or "png", None)

        # Encode images to base64 with the specified format
        image_data = [
            ImageData(b64_json=_encode_image_base64_with_compression(img, format=output_format), revised_prompt=None)
            for img in images
        ]

        response_kwargs = {
            "created": int(time.time()),
            "data": image_data,
            "output_format": output_format,
        }
        if request.size:
            response_kwargs["size"] = size_str
        return ImageGenerationResponse(**response_kwargs)

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Image generation failed: {str(e)}"
        )


@router.post(
    "/v1/images/edits",
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def edit_images(
    raw_request: Request,
    image: list[UploadFile] | None = File(None),
    image_array: list[UploadFile] | None = File(None, alias="image[]"),
    url: list[str] | None = Form(None),
    url_array: list[str] | None = Form(None, alias="url[]"),
    prompt: str = Form(...),
    model: str = Form(None),
    n: int = Form(1),
    size: str = Form("auto"),
    response_format: str = Form("b64_json"),
    output_format: str | None = Form("png"),
    background: str | None = Form("auto"),
    output_compression: Annotated[int, Form(ge=0, le=100)] = 100,
    user: str | None = Form(None),  # unused now
    # vllm-omni extensions for image editing
    mask_image: str | UploadFile | None = None,
    reference_image: str | UploadFile | None = None,
    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    strength: float | None = Form(None),
    true_cfg_scale: float | None = Form(None),
    seed: int | None = Form(None),
    generator_device: str | None = Form(None),
    # vllm-omni extension for per-request LoRA.
    lora: str | None = Form(None),  # Json string
    # vllm-omni extension for layered models (e.g., Qwen-Image-Layered)
    layers: int | None = Form(None),
    resolution: int | None = Form(None),  # See SUPPORTED_LAYERED_RESOLUTIONS
) -> ImageGenerationResponse:
    """
    OpenAI-compatible image edit endpoint.
    """
    # 1. get engine and model
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)
    if model is not None and model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(f"Model mismatch: request specifies '{model}' but server is running '{model_name}'."),
        )
    # 2. get output format & compression
    output_format = _choose_output_format(output_format, background)
    if response_format != "b64_json":
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only response_format 'b64_json' is supported now.",
        )
    try:
        # 2. Build prompt & images params
        prompt: OmniTextPrompt = {"prompt": prompt}
        if negative_prompt is not None:
            prompt["negative_prompt"] = negative_prompt
        input_images_list = []
        images = image or image_array
        urls = url or url_array
        if images:
            input_images_list.extend(images)
        if urls:
            input_images_list.extend(urls)
        if not input_images_list:
            raise HTTPException(status_code=422, detail="Field 'image' or 'url' is required")
        # Reject oversized multi-image edit requests before fetching or decoding
        # any inputs. This keeps over-limit URL requests from burning network,
        # CPU, and memory on work that will be rejected anyway.
        max_input_images = _get_max_edit_input_images(raw_request, engine_client)
        if max_input_images is not None and len(input_images_list) > max_input_images:
            detail = (
                "Received multiple input images. Only a single image is supported by this model."
                if max_input_images == 1
                else (
                    f"Received {len(input_images_list)} input images. "
                    f"At most {max_input_images} images are supported by this model."
                )
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=detail,
            )
        pil_images = await _load_input_images(input_images_list)
        prompt["multi_modal_data"] = {}
        prompt["multi_modal_data"]["image"] = pil_images

        if mask_image is not None:
            loaded = await _load_input_images([mask_image])
            prompt["multi_modal_data"]["mask_image"] = loaded[0]

        if reference_image is not None:
            loaded = await _load_input_images([reference_image])
            prompt["multi_modal_data"]["reference_image"] = loaded[0]

        # 3 Build sample params
        gen_params = OmniDiffusionSamplingParams()
        # 3.0 Init with system default values
        app_state_args = getattr(raw_request.app.state, "args", None)
        default_sample_param = getattr(app_state_args, "default_sampling_params", None)
        # Currently only have one diffusion stage.
        diffusion_stage_ids = [i for i, cfg in enumerate(stage_configs) if get_stage_type(cfg) == "diffusion"]
        if not diffusion_stage_ids:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                detail="No diffusion stage found in multi-stage pipeline.",
            )
        diffusion_stage_id = diffusion_stage_ids[0]
        apply_stage_default_sampling_params(
            default_sample_param,
            gen_params,
            str(diffusion_stage_id),
        )
        _update_if_not_none(gen_params, "num_outputs_per_prompt", n)
        # 3.1 Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        lora_dict = _get_lora_from_json_str(lora)
        lora_request, lora_scale = _parse_lora_request(lora_dict)
        _update_if_not_none(gen_params, "lora_request", lora_request)
        _update_if_not_none(gen_params, "lora_scale", lora_scale)
        # 3.2 Validate resolution if provided
        if resolution is not None and resolution not in SUPPORTED_LAYERED_RESOLUTIONS:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Invalid resolution {resolution}. Supported resolutions: {SUPPORTED_LAYERED_RESOLUTIONS}.",
            )
        # 3.2.1 Validate layers if provided
        try:
            layers = validate_layered_layers(layers)
        except ValueError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=str(e),
            ) from e
        # 3.2.2 Check for conflicting size and resolution parameters
        if resolution is not None and size.lower() != "auto":
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Cannot specify both 'resolution' and 'size'. "
                "Use 'resolution' with size='auto', or use 'size' without 'resolution'.",
            )

        # 3.3 Parse and add size if provided
        width, height = None, None
        if size.lower() == "auto":
            if resolution is None:
                # No resolution specified, use input image size
                width, height = pil_images[0].size
            # else: let pipeline calculate dimensions based on resolution
        else:
            width, height = parse_size(size)

        _check_max_generated_image_size(app_state_args, width, height, resolution)

        size_str = f"{width}x{height}" if width is not None and height is not None else "auto"

        # Keep AR stage target grid in sync with requested output size.
        # GLM-Image consumes target_h/target_w via mm_processor_kwargs.
        if width is not None and height is not None:
            prompt["mm_processor_kwargs"] = {
                "target_h": height,
                "target_w": width,
            }
            # Backward-compatible fallback for processors reading top-level fields.
            prompt["height"] = height
            prompt["width"] = width

        _update_if_not_none(gen_params, "width", width)
        _update_if_not_none(gen_params, "height", height)

        # 3.4 Add optional parameters ONLY if provided
        _update_if_not_none(gen_params, "num_inference_steps", num_inference_steps)
        _update_if_not_none(gen_params, "guidance_scale", guidance_scale)
        _update_if_not_none(gen_params, "strength", strength)
        _update_if_not_none(gen_params, "true_cfg_scale", true_cfg_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        _update_if_not_none(gen_params, "seed", seed if seed is not None else random.randint(0, MAX_UINT32_SEED))
        _update_if_not_none(gen_params, "generator_device", generator_device)
        _update_if_not_none(gen_params, "layers", layers)
        _update_if_not_none(gen_params, "resolution", resolution)

        # 4. Generate images
        request_id = f"img_edit-{random_uuid()}"
        raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
        logger.info(f"Generating {n} image(s) {size_str}")

        if len(stage_configs) > 1:
            # Multi-stage pipeline (e.g. GLM-Image AR+Diffusion): route through
            # the chat handler so the AR stage gets correct max_tokens and
            # target_h/w (same path as /v1/images/generations).
            chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_handler is None:
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="openai_serving_chat is not initialized for multi-stage image editing.",
                )

            # Encode input images to base64 for generate_diffusion_images.
            import base64
            import io as _io

            ref_b64_list: list[str] = []
            for _img in pil_images:
                buf = _io.BytesIO()
                _img.save(buf, format="PNG")
                ref_b64_list.append(base64.b64encode(buf.getvalue()).decode())

            effective_seed = seed if seed is not None else random.randint(0, MAX_UINT32_SEED)
            extra_body: dict[str, Any] = {
                "seed": effective_seed,
                "num_outputs_per_prompt": n,
            }
            if width is not None:
                extra_body["width"] = width
            if height is not None:
                extra_body["height"] = height
            if negative_prompt is not None:
                extra_body["negative_prompt"] = negative_prompt
            if num_inference_steps is not None:
                extra_body["num_inference_steps"] = num_inference_steps
            if guidance_scale is not None:
                extra_body["guidance_scale"] = guidance_scale
            if strength is not None:
                extra_body["strength"] = strength
            if true_cfg_scale is not None:
                extra_body["true_cfg_scale"] = true_cfg_scale
            if layers is not None:
                extra_body["layers"] = layers
            if resolution is not None:
                extra_body["resolution"] = resolution
            if lora is not None:
                # Validate LoRA, then pass through.
                lora_dict = _get_lora_from_json_str(lora)
                _parse_lora_request(lora_dict)
                extra_body["lora"] = lora_dict

            prompt_text = prompt.get("prompt", "")
            generation_result = await chat_handler.generate_diffusion_images(
                prompt=prompt_text,
                extra_body=extra_body,
                reference_images=ref_b64_list,
                request_id=request_id,
            )
            if isinstance(generation_result, ErrorResponse):
                raise HTTPException(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    detail=generation_result.message,
                )
            images, _, _ = generation_result
        else:
            # Single-stage diffusion: use the direct path.
            result = await _generate_with_async_omni(
                engine_client=engine_client,
                gen_params=gen_params,
                stage_configs=stage_configs,
                prompt=prompt,
                request_id=request_id,
            )
            images = _extract_images_from_result(result)

        logger.info(f"Successfully generated {len(images)} image(s)")

        # Encode images to base64
        image_data = [
            ImageData(
                b64_json=_encode_image_base64_with_compression(
                    img, format=output_format, output_compression=output_compression
                ),
                revised_prompt=None,
            )
            for img in images
        ]

        return ImageGenerationResponse(
            created=int(time.time()),
            data=image_data,
            output_format=output_format,
            size=size_str,
        )

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image edit failed: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Image edit failed: {str(e)}")


def _get_engine_and_model(raw_request: Request):
    # Get engine client (AsyncOmni) from app state
    engine_client: EngineClient | AsyncOmni | None = getattr(raw_request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Multi-stage engine not initialized. Start server with a multi-stage omni model.",
        )

    # Check if there's a diffusion stage.
    # Prefer app state (compat layer populated at startup), then fall back to
    # the engine client's stage configs for refactored AsyncOmni paths.
    stage_configs = getattr(raw_request.app.state, "stage_configs", None)
    if not stage_configs:
        stage_configs = getattr(engine_client, "stage_configs", None)
    if not stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )

    normalized_stage_configs = list(stage_configs)
    has_diffusion_stage = any(get_stage_type(stage_cfg) == "diffusion" for stage_cfg in normalized_stage_configs)

    if not has_diffusion_stage:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="No diffusion stage found in multi-stage pipeline.",
        )

    # Get server's loaded model name
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    base_model_paths = getattr(serving_models, "base_model_paths", None) if serving_models else None
    if base_model_paths:
        model_name = base_model_paths[0].name
    else:
        model_name = "unknown"

    return engine_client, model_name, normalized_stage_configs


def _get_diffusion_od_config(raw_request: Request, engine_client: Any) -> Any:
    diffusion_engine = getattr(raw_request.app.state, "diffusion_engine", None) or engine_client
    get_diffusion_od_config = getattr(diffusion_engine, "get_diffusion_od_config", None)
    return (
        get_diffusion_od_config() if callable(get_diffusion_od_config) else getattr(diffusion_engine, "od_config", None)
    )


def _get_max_edit_input_images(raw_request: Request, engine_client: Any) -> int | None:
    od_config = _get_diffusion_od_config(raw_request, engine_client)
    if od_config is None:
        # Preserve the existing compatibility behavior when the diffusion
        # config is not exposed on the serving surface.
        return None

    supports_multimodal_inputs = getattr(od_config, "supports_multimodal_inputs", None)
    if not isinstance(supports_multimodal_inputs, bool):
        # Older serving surfaces and mocked engines may expose a placeholder
        # object instead of a real diffusion config. Treat that as "unknown"
        # so existing single-image flows keep working.
        return None

    if not supports_multimodal_inputs:
        return 1

    max_input_images = getattr(od_config, "max_multimodal_image_inputs", None)
    if max_input_images is None:
        return None
    if isinstance(max_input_images, bool) or not isinstance(max_input_images, Integral):
        return None
    if max_input_images < 1:
        return None
    return int(max_input_images)


def _get_lora_from_json_str(lora_body):
    if lora_body is None:
        return None
    try:
        lora_dict = json.loads(lora_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid LoRA JSON string")

    if not isinstance(lora_dict, dict):
        raise HTTPException(status_code=400, detail="LoRA must be a JSON object")

    return lora_dict


def _parse_lora_request(lora_body: dict[str, Any]):
    try:
        return parse_lora_request(lora_body)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=str(e),
        ) from e


async def _generate_with_async_omni(
    engine_client: AsyncOmni | Any,
    gen_params: Any,
    stage_configs: list[Any],
    **kwargs,
):
    engine_client = cast(AsyncOmni, engine_client)
    result = None
    normalized_stage_configs = list(stage_configs)
    if not normalized_stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )
    sampling_params_list = build_stage_sampling_params_list(
        normalized_stage_configs,
        get_default_sampling_params_list(engine_client),
        diffusion_params=gen_params,
        replace_diffusion_params=True,
    )

    async for output in engine_client.generate(
        sampling_params_list=sampling_params_list,
        **kwargs,
    ):
        result = output

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="No output generated from multi-stage pipeline.",
        )
    return result


def _check_max_generated_image_size(
    app_state_args: Any,
    width: int | None,
    height: int | None,
    resolution: int | None = None,
) -> None:
    """Raise 400 if the requested image size exceeds --max-generated-image-size."""
    max_generated_image_size = getattr(app_state_args, "max_generated_image_size", None)
    # Check max_generated_image_size
    if max_generated_image_size is None:
        return
    if width is not None and height is not None:
        if width * height > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested image size {width}x{height} exceeds the maximum allowed "
                f"size of {max_generated_image_size} pixels.",
            )
    elif resolution is not None:
        # When resolution is set, the output size is resolution * resolution
        if resolution * resolution > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested resolution {resolution} (max {resolution}x{resolution} pixels) "
                f"exceeds the maximum allowed size of {max_generated_image_size} pixels.",
            )


def _update_if_not_none(object: Any, key: str, val: Any) -> None:
    if val is not None:
        setattr(object, key, val)


def _normalize_image(image: Any) -> Any:
    """Normalize a single image output to a PIL-compatible format."""
    if isinstance(image, Image.Image):
        return image
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Unsupported dtype: {image.dtype}")
    if isinstance(image, np.ndarray):
        while image.ndim > 3:
            image = image[0]
        if image.min() < 0:
            if image.min() < -1.01 or image.max() > 1.01:
                logger.warning(
                    f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [-1, 1]. "
                    f"Clipping to [-1, 1] before normalization."
                )
            image = np.clip(image, -1.0, 1.0) * 0.5 + 0.5
        elif image.max() > 1.01:
            logger.warning(
                f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [0, 1]. "
                f"Clipping to [0, 1] before normalization."
            )
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(image)
    return image


def _extract_images_from_result(result: Any) -> list[Any]:
    images = []
    if hasattr(result, "images") and result.images:
        images = result.images
    elif hasattr(result, "request_output"):
        request_output = result.request_output
        if isinstance(request_output, dict) and request_output.get("images"):
            images = request_output["images"]
        elif hasattr(request_output, "images") and request_output.images:
            images = request_output.images
    # Handle when generate more than one image
    if images and isinstance(images[0], np.ndarray) and images[0].shape[0] > 1 and images[0].ndim == 5:
        # Unwrap batch: (N, T, H, W, C) -> [img1, img2, ...]
        images = list(images[0])
    # Flatten nested lists (e.g., from layered models like Qwen-Image-Layered).
    # Note: This only flattens one level deep. Deeper nesting is not supported.
    flattened = []
    for img in images:
        if isinstance(img, list):
            flattened.extend(img)
        else:
            flattened.append(img)
    return [_normalize_image(img) for img in flattened]


async def _load_input_images(
    inputs: list[str],
) -> list[Image.Image]:
    """
    convert to PIL.Image.Image list
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    images: list[Image.Image] = []

    for inp in inputs:
        # 1. URL + base64
        if isinstance(inp, str) and inp.startswith("data:image"):
            try:
                _, b64_data = inp.split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {e}")

        # 2. URL
        elif isinstance(inp, str) and inp.startswith("http"):
            async with httpx.AsyncClient(timeout=60) as client:
                try:
                    resp = await client.get(inp)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content))
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to download image from URL {inp}: {e}")

        # 3. UploadFile
        elif hasattr(inp, "file"):
            try:
                img_data = await inp.read()
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            except Exception as e:
                raise ValueError(f"Failed to open uploaded file: {e}")
        else:
            raise ValueError(f"Unsupported input: {inp}")

    if not images:
        raise ValueError("No valid input images found")

    return images


def _choose_output_format(output_format: str | None, background: str | None) -> str:
    # Normalize and choose extension
    fmt = (output_format or "").lower()
    if fmt in {"jpg", "png", "webp", "jpeg"}:
        return fmt
    # If transparency requested, prefer png
    if (background or "auto").lower() == "transparent":
        return "png"
    # Default
    return "jpeg"


def _prepare_image_for_output_format(image: Image.Image, format: str) -> Image.Image:
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg"}:
        return image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        flattened = Image.new("RGB", alpha_image.size, (255, 255, 255))
        flattened.paste(alpha_image, mask=alpha_image.getchannel("A"))
        return flattened

    return image.convert("RGB")


def _encode_image_base64_with_compression(
    image: Image.Image, format: str = "png", output_compression: int = 100
) -> str:
    """Encode PIL Image to a base64 image string.

    Args:
        image: PIL Image object
        format: Output image format (e.g., "PNG", "JPEG", "WEBP")
        output_compression: Compression level (0-100%), 100 for best quality
    Returns:
        Base64-encoded image as string
    """
    buffer = io.BytesIO()
    image = _prepare_image_for_output_format(image, format)
    save_kwargs = {}
    if format in ("jpg", "jpeg", "webp"):
        save_kwargs["quality"] = output_compression
    elif format == "png":
        save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))  # Map 0-100 to 9-0

    image.save(buffer, format=format, **save_kwargs)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def apply_stage_default_sampling_params(
    default_params_json: str | None,
    sampling_params: Any,
    stage_key: str,
) -> None:
    """
    Update a stage's sampling parameters with vLLM-Omni defaults.

    Args:
        default_params_json: JSON string of stage-keyed default parameters
        sampling_params: The sampling parameters object to update
        stage_key: The stage ID/key in the pipeline
    """
    if default_params_json is not None:
        default_params_dict = json.loads(default_params_json)
        if stage_key in default_params_dict:
            stage_defaults = default_params_dict[stage_key]
            for param_name, param_value in stage_defaults.items():
                if hasattr(sampling_params, param_name):
                    setattr(sampling_params, param_name, param_value)


def _resolve_video_runtime_context(raw_request: Request) -> tuple[str | None, list[Any] | None]:
    app_model_name = None
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    if serving_models and getattr(serving_models, "base_model_paths", None):
        base_paths = serving_models.base_model_paths
        if base_paths:
            app_model_name = base_paths[0].name

    app_stage_configs = getattr(raw_request.app.state, "stage_configs", None)
    return app_model_name, app_stage_configs


def _parse_form_json(value: str | None, expected_type: type | None = None) -> Any:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid JSON in form field.",
        ) from exc
    if expected_type is not None and not isinstance(parsed, expected_type):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid JSON in form field: expected {expected_type.__name__}, got {type(parsed).__name__}.",
        )
    return parsed


def video_response_from_request(model_name: str, req: VideoGenerationRequest) -> VideoResponse:
    resp = VideoResponse(
        model=model_name,
        status=VideoGenerationStatus.QUEUED,
        size=req.size,
        prompt=req.prompt,
    )
    resp.seconds = str(req.seconds or resp.seconds)
    return resp


def _status_code_for_video_failure(error: VideoError | None) -> int:
    if error is None:
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if isinstance(error.code, int):
        if 400 <= error.code < 600:
            return error.code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "HTTPException":
        status_text, _, _ = error.message.partition(":")
        try:
            status_code = int(status_text)
        except ValueError:
            return HTTPStatus.INTERNAL_SERVER_ERROR.value
        if 400 <= status_code < 600:
            return status_code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "EngineDeadError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value
    if error.code == "EngineGenerateError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    return HTTPStatus.INTERNAL_SERVER_ERROR.value


def _video_error_from_exception(exc: Exception) -> VideoError:
    if isinstance(exc, HTTPException):
        message = str(exc.detail) if exc.detail else str(exc)
        return VideoError(code=exc.status_code, message=message)

    if isinstance(exc, (EngineGenerateError, EngineDeadError)):
        err = create_error_response(exc)
        return VideoError(code=err.error.code, message=err.error.message)

    return VideoError(
        code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        message=str(exc),
    )


def _cleanup_video(video_id: str, output_path: str | None):
    try:
        if output_path is not None:
            os.remove(output_path)
    except OSError:
        logger.warning("Failed to cleanup partial video file '%s' for id=%s", output_path, video_id)


async def _run_video_generation_job(
    handler: OmniOpenAIServingVideo,
    request: VideoGenerationRequest,
    video_id: str,
    reference_image: ReferenceImage | None = None,
    app_state: Any | None = None,
) -> None:
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        logger.warning("Video job %s missing before generation task started; skipping", video_id)
        return

    await VIDEO_STORE.update_fields(video_id, {"status": VideoGenerationStatus.IN_PROGRESS})
    started_at = time.perf_counter()
    output_path = None
    try:
        video_bytes, stage_durations, peak_memory_mb = await handler.generate_video_bytes(
            request, video_id, reference_image=reference_image
        )

        file_name = f"{video_id}.{job.file_extension}"
        output_path = await STORAGE_MANAGER.save(video_bytes, file_name)
        logger.info("Video request %s persisted %s output file.", video_id, output_path)

        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.COMPLETED,
                "progress": 100,
                "file_name": file_name,
                "completed_at": int(time.time()),
                "inference_time_s": time.perf_counter() - started_at,
                "stage_durations": stage_durations,
                "peak_memory_mb": peak_memory_mb,
            },
        )
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Video generation failed (engine error) for id=%s", video_id)

        _cleanup_video(video_id, output_path)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
        # Background tasks can't propagate exceptions to FastAPI handlers.
        # Actively signal shutdown when the engine is dead.
        if app_state is not None and isinstance(exc, EngineDeadError):
            terminate_if_errored(
                server=app_state.server,
                engine=app_state.engine_client,
            )
    except Exception as exc:
        logger.exception("Video generation failed for id=%s", video_id)

        _cleanup_video(video_id, output_path)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
    except asyncio.CancelledError:
        _cleanup_video(video_id, output_path)
        await VIDEO_STORE.pop(video_id)
        raise


VIDEO_SYNC_TIMEOUT_S = 600.0


async def _parse_video_form(
    raw_request: Request,
    prompt: str = Form(...),
    input_reference: UploadFile | None = File(default=None),
    image_reference: str | None = Form(default=None),
    model: str | None = Form(default=None),
    seconds: SecondStr | None = Form(default=None),
    size: SizeStr | None = Form(default=None),
    user: str | None = Form(default=None),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    num_frames: int | None = Form(default=None),
    fps: int | None = Form(default=None),
    num_inference_steps: int | None = Form(default=None),
    guidance_scale: float | None = Form(default=None),
    guidance_scale_2: float | None = Form(default=None),
    boundary_ratio: float | None = Form(default=None),
    flow_shift: float | None = Form(default=None),
    true_cfg_scale: float | None = Form(default=None),
    seed: int | None = Form(default=None),
    negative_prompt: str | None = Form(default=None),
    enable_frame_interpolation: bool | None = Form(default=None),
    frame_interpolation_exp: int | None = Form(default=None, ge=1),
    frame_interpolation_scale: float | None = Form(default=None, gt=0.0),
    frame_interpolation_model_path: str | None = Form(default=None),
    lora: str | None = Form(default=None),
    extra_params: str | None = Form(default=None),
) -> tuple[VideoGenerationRequest, "OmniOpenAIServingVideo", str, ReferenceImage | None]:
    """FastAPI dependency that parses video form data, validates inputs,
    resolves the handler, and decodes any reference image.

    Used by both ``POST /v1/videos`` (async) and ``POST /v1/videos/sync``.
    """
    input_reference_bytes = await input_reference.read() if input_reference is not None else None
    parsed_image_reference = _parse_form_json(image_reference)

    if parsed_image_reference is not None and input_reference_bytes is not None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Provide either input_reference or image_reference, not both.",
        )

    request_data: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "seconds": seconds,
        "size": size,
        "image_reference": parsed_image_reference,
        "user": user,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "guidance_scale_2": guidance_scale_2,
        "boundary_ratio": boundary_ratio,
        "flow_shift": flow_shift,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "enable_frame_interpolation": enable_frame_interpolation,
        "frame_interpolation_exp": frame_interpolation_exp,
        "frame_interpolation_scale": frame_interpolation_scale,
        "frame_interpolation_model_path": frame_interpolation_model_path,
        "lora": _parse_form_json(lora, expected_type=dict),
        "extra_params": _parse_form_json(extra_params, expected_type=dict),
    }
    request_data = {k: v for k, v in request_data.items() if v is not None}
    request = VideoGenerationRequest(**request_data)

    handler = Omnivideo(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Video generation handler not initialized.",
        )
    logger.info("Video generation handler: %s", type(handler).__name__)
    try:
        app_model_name, app_stage_configs = _resolve_video_runtime_context(raw_request)
        effective_model_name = handler.model_name or app_model_name or request.model or "unknown"
        if request.model is not None and effective_model_name is not None and request.model != effective_model_name:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    f"Model mismatch: request specifies '{request.model}' but server is running "
                    f"'{effective_model_name}'."
                ),
            )
        handler.set_stage_configs_if_missing(app_stage_configs)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video generation setup failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation setup failed: {str(e)}",
        )

    try:
        image_data = await decode_input_reference(request.image_reference, input_reference_bytes)
    except InvalidInputReferenceError as exc:
        raise HTTPException(400, detail=str(exc) or "Invalid input reference.") from exc

    reference_image = ReferenceImage(data=image_data) if image_data is not None else None
    return request, handler, effective_model_name, reference_image


@router.post(
    "/v1/videos",
    responses={
        HTTPStatus.OK.value: {"model": VideoResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def create_video(
    raw_request: Request,
    ctx: tuple[VideoGenerationRequest, OmniOpenAIServingVideo, str, ReferenceImage | None] = Depends(_parse_video_form),
) -> VideoResponse:
    """Create an asynchronous video generation job.

    Accepts multipart form-data (see ``_parse_video_form`` for parameters),
    persists a queued job record, and starts generation in the background.
    """
    request, handler, effective_model_name, reference_image = ctx
    ref = video_response_from_request(effective_model_name, request)
    await VIDEO_STORE.upsert(ref.id, ref)
    task = asyncio.create_task(
        _run_video_generation_job(handler, request, ref.id, reference_image, app_state=raw_request.app.state)
    )
    await VIDEO_TASKS.upsert(ref.id, task)
    return ref


@router.post(
    "/v1/videos/sync",
    responses={
        HTTPStatus.OK.value: {"content": {"video/mp4": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def create_video_sync(
    raw_request: Request,
    ctx: tuple[VideoGenerationRequest, OmniOpenAIServingVideo, str, ReferenceImage | None] = Depends(_parse_video_form),
) -> Response:
    """Synchronous video generation endpoint.

    Accepts the same form parameters as ``POST /v1/videos`` but blocks until
    generation completes and returns raw video bytes (``video/mp4``) directly.
    Designed for benchmark and testing scenarios.

    Metadata is returned via response headers ``X-Request-Id``,
    ``X-Model``, and ``X-Inference-Time-S``.
    """
    request, handler, effective_model_name, reference_image = ctx
    request_id = f"video_sync-{random_uuid()}"
    raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
    started_at = time.perf_counter()
    try:
        video_bytes, stage_durations, peak_memory_mb = await asyncio.wait_for(
            handler.generate_video_bytes(request, request_id, reference_image=reference_image),
            timeout=VIDEO_SYNC_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
            detail=f"Video generation timed out after {VIDEO_SYNC_TIMEOUT_S}s.",
        )
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Sync video generation failed for request_id=%s", request_id)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation failed: {str(exc)}",
        ) from exc
    inference_time_s = time.perf_counter() - started_at

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "X-Request-Id": request_id,
            "X-Model": effective_model_name,
            "X-Inference-Time-S": f"{inference_time_s:.3f}",
            "X-Stage-Durations": json.dumps(stage_durations, separators=(",", ":")),
            "X-Peak-Memory-MB": f"{peak_memory_mb:.3f}",
        },
    )


@router.get("/v1/videos", response_model=VideoListResponse)
async def list_videos(
    after: str | None = None,
    limit: int | None = Query(None, ge=0, le=100),
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
):
    """List stored video generation jobs.

    Args:
        after: Optional cursor indicating the last seen video ID.
        limit: Optional maximum number of jobs to return.
        order: Sort order for the returned jobs by creation time.

    Returns:
        A ``VideoListResponse`` containing paginated job metadata and cursor
        information.
    """
    jobs = await VIDEO_STORE.list_values()
    jobs.sort(key=lambda j: j.created_at, reverse=order == "desc")

    if after is not None:
        idx = next((i for i, job in enumerate(jobs) if job.id == after), None)
        jobs = [] if idx is None else jobs[idx + 1 :]

    has_more = False
    if limit is not None:
        has_more = len(jobs) > limit
        jobs = jobs[:limit]

    first_id, last_id = None, None
    if len(jobs) > 0:
        first_id = jobs[0].id
        last_id = jobs[-1].id

    return VideoListResponse(data=jobs, has_more=has_more, first_id=first_id, last_id=last_id)


@router.get("/v1/videos/{video_id}", response_model=None)
async def retrieve_video(video_id: str) -> VideoResponse | JSONResponse:
    """Retrieve metadata for a previously created video job.

    Args:
        video_id: Identifier returned by ``POST /v1/videos``.

    Returns:
        The stored ``VideoResponse`` for the requested job.

    Raises:
        HTTPException: If the video job does not exist.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if job.status == VideoGenerationStatus.FAILED:
        status_code = _status_code_for_video_failure(job.error)
        content = job.model_dump(mode="json")
        if content.get("error") is not None:
            content["error"]["code"] = status_code
        return JSONResponse(
            content=content,
            status_code=status_code,
        )
    return job


@router.delete("/v1/videos/{video_id}")
async def delete_video(video_id: str) -> VideoDeleteResponse:
    """Delete a stored video job and any generated output.

    If the job is still queued or running, this endpoint first attempts to
    cancel the in-flight generation task before removing the stored metadata.

    Args:
        video_id: Identifier of the video job to delete.

    Returns:
        A ``VideoDeleteResponse`` confirming the job was removed.

    Raises:
        HTTPException: If the video job does not exist, cancellation is still
        in progress, or output is not yet ready for a completed job.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.status in (VideoGenerationStatus.QUEUED, VideoGenerationStatus.IN_PROGRESS):
        task = await VIDEO_TASKS.get(video_id)
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=409, detail="Cancellation in progress. Please try again later.")
            except asyncio.CancelledError:
                pass

            await VIDEO_STORE.pop(video_id)
            return VideoDeleteResponse(id=job.id, deleted=True)
    elif job.status is VideoGenerationStatus.FAILED:
        if job.file_name is not None:
            try:
                await STORAGE_MANAGER.delete(job.file_name)
            except Exception:
                logger.warning("Failed to delete stored artifact for failed video job %s", video_id, exc_info=True)

        await VIDEO_STORE.pop(video_id)
        return VideoDeleteResponse(id=job.id, deleted=True)

    if job.file_name is None:
        raise HTTPException(status_code=409, detail="Video output not yet available. Please try again later.")

    await STORAGE_MANAGER.delete(job.file_name)
    await VIDEO_STORE.pop(video_id)
    return VideoDeleteResponse(id=job.id, deleted=True)


@router.get("/v1/videos/{video_id}/content")
async def download_video(video_id: str) -> FileResponse:
    """Download the generated file for a completed video job.

    Args:
        video_id: Identifier of the video job whose output should be returned.

    Returns:
        A ``FileResponse`` streaming the generated video file from local
        storage.

    Raises:
        HTTPException: If the job does not exist, is still in progress, or the
        generated file is missing from disk.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.status == VideoGenerationStatus.FAILED:
        raise HTTPException(status_code=422, detail="Video generation failed. Check job status for error details.")
    if not job.file_name:
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    full_path = STORAGE_MANAGER.get_full_file_path(job.file_name)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Generated video file not found on disk")

    return FileResponse(path=full_path, media_type=job.media_type, filename=job.file_name)


@profiler_router.post("/start_profile")
async def start_profile(raw_request: Request, request: ProfileRequest | None = None):
    """Start profiling for the engine.

    Args:
        request: Optional request body with stages to profile.
            - stages: List of stage IDs to profile. If None, profiles all stages.

    Example:
        POST /start_profile
        {"stages": [0, 1]}  # Profile only stages 0 and 1
    """
    try:
        stages = request.stages if request else None
        logger.info("Starting profiler for stages: %s", stages if stages else "all")
        engine_client = raw_request.app.state.engine_client
        result = await engine_client.start_profile(stages=stages)
        logger.info("Profiler started.")
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Failed to start profiler: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Failed to start profiler: {str(e)}"
        )


@profiler_router.post("/stop_profile")
async def stop_profile(raw_request: Request, request: ProfileRequest | None = None):
    """Stop profiling for the engine.

    Args:
        request: Optional request body with stages to stop profiling.
            - stages: List of stage IDs to stop profiling. If None, stops all stages.

    Example:
        POST /stop_profile
        {"stages": [0, 1]}  # Stop profiling only stages 0 and 1
    """
    try:
        stages = request.stages if request else None
        logger.info("Stopping profiler for stages: %s", stages if stages else "all")
        engine_client = raw_request.app.state.engine_client
        result = await engine_client.stop_profile(stages=stages)
        logger.info("Profiler stopped.")
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Failed to stop profiler: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Failed to stop profiler: {str(e)}"
        )


class OmniSleepRequest(BaseModel):
    stage_ids: list[int]
    level: int = 2


class OmniWakeupRequest(BaseModel):
    stage_ids: list[int]


@router.post("/v1/omni/sleep")
async def omni_sleep(request: OmniSleepRequest, raw_request: Request):
    engine_client = raw_request.app.state.engine_client
    sleeping_set = raw_request.app.state.sleeping_stages
    if not hasattr(engine_client, "sleep"):
        raise HTTPException(status_code=501, detail="Engine does not support sleep")
    acks = await engine_client.sleep(stage_ids=request.stage_ids, level=request.level)
    for sid in request.stage_ids:
        sleeping_set.add(sid)
    return {"status": "SUCCESS", "acks": [dataclasses.asdict(a) if dataclasses.is_dataclass(a) else a for a in acks]}


@router.post("/v1/omni/wakeup")
async def omni_wakeup(request: OmniWakeupRequest, raw_request: Request):
    engine_client = raw_request.app.state.engine_client
    sleeping_set = raw_request.app.state.sleeping_stages
    if not any(sid in sleeping_set for sid in request.stage_ids):
        return {"status": "SKIPPED", "reason": "Target stages are not sleeping."}
    if not hasattr(engine_client, "wake_up"):
        raise HTTPException(status_code=501, detail="Engine does not support wake_up")
    acks = await engine_client.wake_up(stage_ids=request.stage_ids)
    for sid in request.stage_ids:
        if sid in sleeping_set:
            sleeping_set.remove(sid)
    return {"status": "SUCCESS", "acks": [dataclasses.asdict(a) if dataclasses.is_dataclass(a) else a for a in acks]}


if __name__ == "__main__":
    import argparse
    import asyncio

    from vllm.entrypoints.openai.cli_args import make_arg_parser

    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    parser = argparse.ArgumentParser(description="vLLM-Omni OpenAI-Compatible REST API server")
    parser = make_arg_parser(parser)
    registered_flags = set()
    for action in parser._actions:
        registered_flags.update(action.option_strings)
    if "--omni" not in registered_flags:
        parser.add_argument("--omni", action="store_true", default=False, help="Enable vLLM-Omni mode.")
    if "--enable-sleep-mode" not in registered_flags:
        parser.add_argument(
            "--enable-sleep-mode", action="store_true", default=False, help="Enable GPU memory pool for sleep mode."
        )
    nullify_stage_engine_defaults(parser)
    args = parser.parse_args()
    if not hasattr(args, "model_tag"):
        setattr(args, "model_tag", args.model)
    if hasattr(args, "model_tag") and args.model_tag is None:
        args.model_tag = args.model
    asyncio.run(omni_run_server(args))
