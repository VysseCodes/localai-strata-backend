#!/usr/bin/env python3
"""LocalAI backend for Strata (https://github.com/Niko1221/Strata).

Strata runs Qwen3.8-Flash-Next on one GPU plus system RAM and ships its own
OpenAI-compatible server (serve/server.py) that owns the chat template, the
reasoning/tool-call parsing and the engine process. This backend does not
re-implement any of that. LoadModel starts Strata's own server as a private
child process, bound to 127.0.0.1 on a free port and protected by a random
API key, and Predict/PredictStream forward LocalAI's chat requests to it and
translate the answer back into LocalAI's ChatDelta replies.

`parameters.model` in the LocalAI model YAML is one of:
  - a folder of GGUF shards, as LocalAI's model gallery installs them: the
    first load prepares the model for this machine with Strata's own setup
    (prepare.py), later loads use what it wrote
  - the run config Strata's setup writes (strata-<model>.json), or the Strata
    folder that holds exactly one, for a model set up by hand

Model YAML `options:` this backend understands (all optional):
  vram_reserve_mib:N   VRAM the engine leaves free for other models (e.g. a
                       TTS model). Overrides the value setup wrote.
  load_timeout:S       seconds to wait for the model to load (default 1800).
  reasoning_effort:E   default thinking level when a request sets none:
                       none, low, medium or high (Strata's default is high).
  family:F, quant:Q    the model in a GGUF folder (read from the file names
                       when not set): qwen/swift/coder/unsloth, IQ2_XS etc.
  strata_home:DIR      where Strata and its data go (default <models>/strata).
  strata_ref:REF       the Strata commit or tag to prepare with.
  prepare_timeout:S    seconds the one-time preparation may take (default 4 h).
  gpu_backend:B        cuda or hip, when the machine has both kinds of GPU.
  engine:mock          Strata's mock engine, for tests only.
"""
from concurrent import futures
import argparse
import http.client
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import backend_pb2
import backend_pb2_grpc

import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from grpc_auth import get_auth_interceptors  # noqa: E402
import prepare  # noqa: E402

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '4'))

DEFAULT_LOAD_TIMEOUT_S = 1800
EFFORTS = ("none", "low", "medium", "high")
# Strata's own range for top_k (serve/runconfig.py: "1..64").
TOP_K_MAX = 64


def log(*args):
    print("[strata]", *args, file=sys.stderr, flush=True)


def parse_options(options):
    """LocalAI's `options:` list ("key:value" strings) as a dict."""
    out = {}
    for opt in options or []:
        if ":" not in opt:
            continue
        key, value = opt.split(":", 1)
        out[key.strip().lower()] = value.strip()
    return out


def validate_options(opts):
    """`opts` checked up front, so a typo fails the load with a clear message."""
    reserve = opts.get("vram_reserve_mib")
    if reserve is not None and not reserve.isdigit():
        raise ValueError(f"vram_reserve_mib must be a whole number of MiB, not {reserve!r}")
    timeout = opts.get("load_timeout")
    if timeout is not None:
        try:
            if float(timeout) <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"load_timeout must be a positive number of seconds, not {timeout!r}") from None
    effort = opts.get("reasoning_effort")
    if effort is not None:
        effort = opts["reasoning_effort"] = effort.lower()
        if effort not in EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {', '.join(EFFORTS)}, not {effort!r}")
    engine = opts.get("engine")
    if engine is not None and engine not in ("strata", "mock"):
        raise ValueError(f"engine must be strata or mock, not {engine!r}")
    family = opts.get("family")
    if family is not None and family not in prepare.FAMILY_TAGS:
        raise ValueError(f"family must be one of {', '.join(prepare.FAMILY_TAGS)}, not {family!r}")
    gpu = opts.get("gpu_backend")
    if gpu is not None and gpu not in ("cuda", "hip", "sycl"):
        raise ValueError(f"gpu_backend must be cuda, hip or sycl, not {gpu!r}")
    for key in ("prepare_timeout",):
        if opts.get(key) is not None:
            try:
                if float(opts[key]) <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(f"{key} must be a positive number of seconds, not {opts[key]!r}") from None
    ref = opts.get("strata_ref")
    if ref is not None and not re.fullmatch(r"[A-Za-z0-9._/-]{1,100}", ref):
        raise ValueError(f"strata_ref must be a commit or tag, not {ref!r}")
    return opts


def find_run_config(path):
    """The Strata run config for `path`: the file itself, or the one
    strata-<model>.json in a folder (setup writes it next to setup.py)."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        found = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.startswith("strata-") and f.endswith(".json")
            and not f.endswith(".shared-settings.json")
            and not f.endswith(".bak")
        )
        if len(found) == 1:
            return found[0]
        if not found:
            raise FileNotFoundError(f"no strata-*.json run config in {path}; run Strata's setup first")
        raise ValueError(
            f"{path} has several run configs ({', '.join(os.path.basename(f) for f in found)}); "
            "point parameters.model at one of them")
    raise FileNotFoundError(f"{path} does not exist; run Strata's setup first")


def set_arg(args, flag, value):
    """`args` with `flag value` set (replaced if present, else appended)."""
    args = list(args)
    if flag in args[:-1]:
        args[args.index(flag) + 1] = str(value)
    else:
        args += [flag, str(value)]
    return args


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Runs as the child's first step, then becomes Strata's server (execv keeps the
# setting): the server gets SIGTERM if this backend dies, so a crashed or
# SIGKILLed backend never leaves the engine holding the GPU. It runs in the new
# process rather than as a preexec_fn: Python code between fork and exec is not
# safe while gRPC's threads are running, and aborted the child now and then.
# If the backend died before prctl took effect, the parent check catches it.
_PDEATHSIG_EXEC = (
    "import ctypes, os, signal, sys\n"
    "try:\n"
    "    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)\n"   # PR_SET_PDEATHSIG
    "except OSError:\n"
    "    pass\n"
    "if os.getppid() != int(os.environ.pop('STRATA_LOCALAI_PARENT')):\n"
    "    sys.exit(1)\n"
    "os.execv(sys.argv[1], sys.argv[1:])\n"
)


def messages_to_dicts(messages):
    """LocalAI's proto Messages as OpenAI chat messages."""
    result = []
    for msg in messages:
        d = {"role": msg.role, "content": msg.content or ""}
        if msg.name:
            d["name"] = msg.name
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.reasoning_content:
            d["reasoning_content"] = msg.reasoning_content
        if msg.tool_calls:
            try:
                calls = json.loads(msg.tool_calls)
            except json.JSONDecodeError:
                calls = None
            if isinstance(calls, list) and calls:
                d["tool_calls"] = calls
                if not d["content"]:
                    d["content"] = None
        result.append(d)
    return result


class StrataError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def grpc_code(status):
    if status in (400, 413, 422):
        return grpc.StatusCode.INVALID_ARGUMENT
    if status in (401, 403):
        return grpc.StatusCode.PERMISSION_DENIED
    if status == 404:
        return grpc.StatusCode.NOT_FOUND
    if status in (429, 503):
        return grpc.StatusCode.UNAVAILABLE
    return grpc.StatusCode.INTERNAL


class StrataServer:
    """Strata's own server, run as a private child process."""

    def __init__(self, ref, options):
        """`ref`: a run config (strata-<model>.json) or, for the mock engine,
        the Strata folder itself."""
        self.options = options
        self.mock = options.get("engine") == "mock"
        if self.mock:
            self.run_config = None
            self.root = ref if os.path.isdir(ref) else os.path.dirname(os.path.abspath(ref))
        else:
            self.run_config = find_run_config(ref)
            self.root = None
        self.port = free_port()
        self.key = secrets.token_urlsafe(32)
        self.proc = None
        self.tmpdir = None

    # -- start / stop ------------------------------------------------------
    def _config_for_launch(self):
        """The run config to start with: setup's file as it is, or a private
        copy carrying the YAML's overrides (setup's file is never edited)."""
        with open(self.run_config, encoding="utf-8-sig") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"{self.run_config} is not a JSON object")
        self.root = cfg.get("cwd") or os.path.dirname(os.path.abspath(self.run_config))
        changed = False
        reserve = self.options.get("vram_reserve_mib")
        if reserve:
            cfg["args"] = set_arg(cfg.get("args") or [], "--vram-reserve-mib", reserve)
            changed = True
        for key in ("open_browser", "api_monitor"):    # nobody looks at those pages here
            if cfg.get(key):
                cfg[key] = False
                changed = True
        if not changed:
            return self.run_config
        self.tmpdir = tempfile.mkdtemp(prefix="strata-localai-")
        os.chmod(self.tmpdir, 0o700)
        path = os.path.join(self.tmpdir, os.path.basename(self.run_config))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        return path

    def start(self, timeout):
        config_path = None if self.mock else self._config_for_launch()
        root = self.root
        server_py = os.path.join(root, "serve", "server.py")
        python = os.path.join(root, ".venv", "bin", "python")
        if not os.path.isfile(server_py):
            raise FileNotFoundError(f"{server_py} not found: is {root} a Strata folder?")
        if not os.path.isfile(python):
            python = sys.executable
            log(f"no Strata .venv in {root}; using {python}")
        cmd = [python, server_py, "--engine", "mock" if self.mock else "strata",
               "--host", "127.0.0.1", "--port", str(self.port)]
        if config_path:
            cmd += ["--config", config_path]
        env = dict(os.environ)
        env["STRATA_API_KEY"] = self.key          # in the environment, not on the command line (ps)
        env.pop("PYTHONPATH", None)               # Strata's venv, not this backend's
        env.pop("VIRTUAL_ENV", None)
        env["STRATA_LOCALAI_PARENT"] = str(os.getpid())
        log("starting:", " ".join(cmd))
        self.proc = subprocess.Popen(
            [python, "-c", _PDEATHSIG_EXEC] + cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=sys.stderr.fileno(), stderr=sys.stderr.fileno(),
            start_new_session=True,
        )
        self._wait_loaded(timeout)

    def _wait_loaded(self, timeout):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(f"Strata's server exited with code {code} while loading (see the log above)")
            try:
                status, body = self.request("GET", "/health", timeout=5)
                info = json.loads(body or b"{}")
                if status == 200 and (info.get("loaded") or self.mock):
                    log(f"ready on 127.0.0.1:{self.port}: model {info.get('model')}, "
                        f"context {info.get('max_context')}")
                    return
                last = f"HTTP {status}: {body[:200]!r}"
            except (OSError, http.client.HTTPException, ValueError) as e:
                last = repr(e)
            time.sleep(2)
        raise TimeoutError(f"Strata did not load within {timeout:.0f} s (last health check: {last})")

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            log("stopping Strata's server")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    # -- HTTP --------------------------------------------------------------
    def _conn(self, timeout):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def request(self, method, path, body=None, timeout=60):
        conn = self._conn(timeout)
        try:
            conn.request(method, path, body=json.dumps(body) if body is not None else None,
                         headers=self._headers())
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def stream_chat(self, body, is_active):
        """Strata's SSE chunks for a streamed chat completion, as dicts."""
        conn = self._conn(timeout=None)
        try:
            conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=self._headers())
            resp = conn.getresponse()
            if resp.status != 200:
                raw = resp.read()
                try:
                    err = json.loads(raw).get("error")
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                except (ValueError, AttributeError):
                    msg = raw.decode("utf-8", "replace")[:500]
                raise StrataError(resp.status, msg or f"HTTP {resp.status}")
            while True:
                if not is_active():
                    log("client went away: cancelling the request")
                    return
                line = resp.readline()
                if not line:
                    return
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue                       # blank separators and ": ping" comments
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    err = chunk["error"]
                    raise StrataError(500, err.get("message") if isinstance(err, dict) else str(err))
                yield chunk
        finally:
            conn.close()                           # closing the socket is how Strata learns to stop


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def __init__(self):
        self.server = None
        self.options = {}
        self.lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", "utf-8"))

    def LoadModel(self, request, context):
        with self.lock:
            try:
                if self.server is not None:
                    self.server.stop()
                    self.server = None
                self.options = validate_options(parse_options(request.Options))
                ref = (request.ModelFile or "").strip()
                if not ref or not os.path.exists(ref):
                    ref = request.Model
                    if request.ModelPath and not os.path.isabs(ref):
                        ref = os.path.join(request.ModelPath, ref)
                timeout = float(self.options.get("load_timeout", DEFAULT_LOAD_TIMEOUT_S))
                if (self.options.get("engine") != "mock" and os.path.isdir(ref)
                        and prepare.gguf_shards(ref)):
                    # Installed from the model gallery: the shards are here; the
                    # machine-specific part of Strata's setup runs on first load.
                    ref = prepare.prepare(ref, self.options, request.ModelPath, request.ContextSize)
                server = StrataServer(ref, self.options)
                try:
                    server.start(timeout)
                except BaseException:
                    server.stop()
                    raise
                self.server = server
                if request.ContextSize:
                    log(f"LocalAI context_size {request.ContextSize}; Strata uses the context its setup chose "
                        "(change it with setup --setup --context N)")
                return backend_pb2.Result(success=True, message="Model loaded successfully")
            except Exception as e:
                log(f"LoadModel failed: {e}")
                return backend_pb2.Result(success=False, message=f"Unexpected {e!r}")

    def Free(self, request, context):
        with self.lock:
            if self.server is not None:
                self.server.stop()
                self.server = None
        return backend_pb2.Result(success=True, message="Model freed")

    def shutdown(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    # -- inference ---------------------------------------------------------
    def _request_body(self, request):
        if request.Messages:
            messages = messages_to_dicts(request.Messages)
        elif request.Prompt:
            # Strata templates the conversation itself; a pre-templated prompt
            # (use_tokenizer_template: false) can only be sent as one message.
            log("got a templated prompt: set `template: use_tokenizer_template: true` in the model YAML")
            messages = [{"role": "user", "content": request.Prompt}]
        else:
            raise StrataError(400, "the request has no messages")
        if request.Images or request.Videos or request.Audios:
            log("this Strata backend is text only: images, video and audio are left out")

        body = {"model": "strata", "messages": messages, "stream": True}
        if request.Tokens > 0:
            body["max_tokens"] = request.Tokens
        if request.Temperature > 0:
            body["temperature"] = request.Temperature
        if 0 < request.TopP <= 1:
            body["top_p"] = request.TopP
        if 0 < request.TopK <= TOP_K_MAX:
            body["top_k"] = request.TopK
        if 0 < request.MinP <= 1:
            body["min_p"] = request.MinP
        if request.PresencePenalty:
            body["presence_penalty"] = request.PresencePenalty
        if request.FrequencyPenalty:
            body["frequency_penalty"] = request.FrequencyPenalty
        if request.Seed > 0:
            body["seed"] = request.Seed
        stops = [s for s in request.StopPrompts if s]
        if stops:
            body["stop"] = stops[:4]
        if request.Tools:
            try:
                tools = json.loads(request.Tools)
            except json.JSONDecodeError:
                tools = None
            if isinstance(tools, list) and tools:
                body["tools"] = tools
                if request.ToolChoice:
                    try:
                        body["tool_choice"] = json.loads(request.ToolChoice)
                    except json.JSONDecodeError:
                        body["tool_choice"] = request.ToolChoice
        if request.Grammar:
            try:
                schema = json.loads(request.Grammar)
            except json.JSONDecodeError:
                schema = None
            if isinstance(schema, dict):
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "response", "schema": schema}}
            else:
                log("ignoring a BNF grammar: Strata takes JSON schemas only")

        # Thinking. LocalAI turns `reasoning_effort` (and the model config's
        # reasoning settings) into the enable_thinking metadata flag; "none"
        # arrives as enable_thinking=false. A plain level wins if it is passed.
        meta = dict(request.Metadata)
        effort = (meta.get("reasoning_effort") or "").lower()
        thinking = (meta.get("enable_thinking") or "").lower()
        if effort in EFFORTS:
            body["reasoning_effort"] = effort
        elif thinking == "false":
            body["reasoning_effort"] = "none"
        elif thinking != "true" and self.options.get("reasoning_effort") in EFFORTS:
            body["reasoning_effort"] = self.options["reasoning_effort"]
        return body

    def _generate(self, request, context, streaming):
        server = self.server
        if server is None or not server.alive():
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "Strata is not loaded" if server is None else "Strata's server stopped; reload the model")
        try:
            body = self._request_body(request)
        except StrataError as e:
            context.abort(grpc_code(e.status), str(e))

        content, reasoning = [], []
        calls = {}                                   # index -> {"id", "name", "arguments"}
        usage = {}
        try:
            for chunk in server.stream_chat(body, context.is_active):
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    tc_protos = []
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", len(calls)))
                        fn = tc.get("function") or {}
                        slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                        tc_protos.append(backend_pb2.ToolCallDelta(
                            index=idx, id=tc.get("id") or "", name=fn.get("name") or "",
                            arguments=fn.get("arguments") or ""))
                    if text:
                        content.append(text)
                    if think:
                        reasoning.append(think)
                    if streaming and (text or think or tc_protos):
                        kw = {}
                        if text:
                            kw["content"] = text
                        if think:
                            kw["reasoning_content"] = think
                        if tc_protos:
                            kw["tool_calls"] = tc_protos
                        yield backend_pb2.Reply(message=text.encode("utf-8"),
                                                chat_deltas=[backend_pb2.ChatDelta(**kw)])
        except StrataError as e:
            context.abort(grpc_code(e.status), str(e))
        except (OSError, http.client.HTTPException) as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, f"lost the connection to Strata: {e}")

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if streaming:
            # Everything was sent as it came; the last reply carries the counts.
            yield backend_pb2.Reply(message=b"", prompt_tokens=prompt_tokens, tokens=completion_tokens,
                                    chat_deltas=[backend_pb2.ChatDelta()])
            return
        text = "".join(content)
        tool_calls = [backend_pb2.ToolCallDelta(index=i, id=c["id"], name=c["name"], arguments=c["arguments"])
                      for i, c in sorted(calls.items())]
        yield backend_pb2.Reply(
            message=text.encode("utf-8"), prompt_tokens=prompt_tokens, tokens=completion_tokens,
            chat_deltas=[backend_pb2.ChatDelta(content=text, reasoning_content="".join(reasoning),
                                               tool_calls=tool_calls)])

    def Predict(self, request, context):
        return next(self._generate(request, context, streaming=False))

    def PredictStream(self, request, context):
        yield from self._generate(request, context, streaming=True)

    def TokenizeString(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Strata tokenizes inside its own server")


def serve(address):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ('grpc.max_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ],
        interceptors=get_auth_interceptors(),
    )
    servicer = BackendServicer()
    backend_pb2_grpc.add_BackendServicer_to_server(servicer, server)
    server.add_insecure_port(address)
    server.start()
    log("Server started. Listening on: " + address)

    def stop(signum, frame):
        log(f"signal {signum}: shutting down")
        servicer.shutdown()
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    finally:
        servicer.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Strata gRPC backend.")
    parser.add_argument("--addr", default="localhost:50051", help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
