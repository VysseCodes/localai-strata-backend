"""Tests for the Strata backend.

The unit tests need nothing but this backend. The end-to-end tests run
Strata's own server with its mock engine (no GPU, no model download): point
STRATA_DIR at a Strata checkout whose .venv setup.sh has created, or that has
Strata's requirements.txt installed in its .venv.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest

import grpc

import backend_pb2
import backend_pb2_grpc
import backend
import prepare

STRATA_DIR = os.environ.get("STRATA_DIR", "")


class TestHelpers(unittest.TestCase):
    def test_parse_options(self):
        self.assertEqual(
            backend.parse_options(["vram_reserve_mib:6500", "Load_Timeout: 900 ", "junk"]),
            {"vram_reserve_mib": "6500", "load_timeout": "900"})

    def test_validate_options(self):
        self.assertEqual(backend.validate_options({"reasoning_effort": "NONE"})["reasoning_effort"], "none")
        for bad in ({"vram_reserve_mib": "6.5G"}, {"load_timeout": "soon"}, {"load_timeout": "-1"},
                    {"reasoning_effort": "max"}, {"engine": "vllm"}):
            with self.assertRaises(ValueError, msg=bad):
                backend.validate_options(dict(bad))

    def test_set_arg(self):
        self.assertEqual(backend.set_arg(["-c", "4096"], "--vram-reserve-mib", 6500),
                         ["-c", "4096", "--vram-reserve-mib", "6500"])
        self.assertEqual(backend.set_arg(["--vram-reserve-mib", "700", "-x"], "--vram-reserve-mib", "6500"),
                         ["--vram-reserve-mib", "6500", "-x"])

    def test_find_run_config(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                backend.find_run_config(d)
            cfg = os.path.join(d, "strata-iq2_xs.json")
            for name in ("strata-iq2_xs.json", "strata-iq2_xs.shared-settings.json", "strata-iq2_xs.json.bak"):
                with open(os.path.join(d, name), "w") as f:
                    f.write("{}")
            self.assertEqual(backend.find_run_config(d), cfg)
            self.assertEqual(backend.find_run_config(cfg), cfg)
            with open(os.path.join(d, "strata-q2_0.json"), "w") as f:
                f.write("{}")
            with self.assertRaises(ValueError):
                backend.find_run_config(d)

    def test_config_overrides_go_to_a_private_copy(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "strata-iq2_xs.json")
            original = {"exe": "x", "args": ["--vram-reserve-mib", "700"], "cwd": d, "open_browser": True}
            with open(path, "w") as f:
                json.dump(original, f)
            srv = backend.StrataServer(path, {"vram_reserve_mib": "1500"})
            launch = srv._config_for_launch()
            try:
                self.assertNotEqual(launch, path)
                with open(launch) as f:
                    cfg = json.load(f)
                self.assertEqual(cfg["args"], ["--vram-reserve-mib", "1500"])
                self.assertFalse(cfg["open_browser"])
                with open(path) as f:
                    self.assertEqual(json.load(f), original)    # setup's file is untouched
            finally:
                srv.stop()
            self.assertFalse(os.path.exists(launch))
            srv = backend.StrataServer(path, {})
            original.pop("open_browser")
            with open(path, "w") as f:
                json.dump(original, f)
            self.assertEqual(srv._config_for_launch(), path)    # nothing to override: used as it is

    def test_messages_to_dicts(self):
        msgs = [
            backend_pb2.Message(role="system", content="be brief"),
            backend_pb2.Message(role="assistant", content="", tool_calls=json.dumps(
                [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}])),
            backend_pb2.Message(role="tool", content="42", tool_call_id="c1", name="f"),
        ]
        out = backend.messages_to_dicts(msgs)
        self.assertEqual(out[0], {"role": "system", "content": "be brief"})
        self.assertIsNone(out[1]["content"])
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], "{\"a\": 1}")  # stays a string
        self.assertEqual(out[2], {"role": "tool", "content": "42", "tool_call_id": "c1", "name": "f"})

    def test_request_body_thinking(self):
        s = backend.BackendServicer()
        base = dict(Messages=[backend_pb2.Message(role="user", content="hi")])
        body = s._request_body(backend_pb2.PredictOptions(**base, Metadata={"enable_thinking": "false"}))
        self.assertEqual(body["reasoning_effort"], "none")
        body = s._request_body(backend_pb2.PredictOptions(**base, Metadata={"enable_thinking": "true"}))
        self.assertNotIn("reasoning_effort", body)
        body = s._request_body(backend_pb2.PredictOptions(**base, Metadata={"reasoning_effort": "low"}))
        self.assertEqual(body["reasoning_effort"], "low")
        s.options = {"reasoning_effort": "medium"}
        self.assertEqual(s._request_body(backend_pb2.PredictOptions(**base))["reasoning_effort"], "medium")

    def test_request_body_sampling_and_tools(self):
        s = backend.BackendServicer()
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        body = s._request_body(backend_pb2.PredictOptions(
            Messages=[backend_pb2.Message(role="user", content="hi")], Tokens=64, Temperature=0.7, TopP=0.9,
            TopK=200, MinP=0.05, StopPrompts=["", "END"], Tools=json.dumps(tools), ToolChoice="auto",
            Grammar="root ::= \"x\""))
        self.assertEqual(body["max_tokens"], 64)
        self.assertAlmostEqual(body["temperature"], 0.7, places=5)
        self.assertNotIn("top_k", body)                    # outside Strata's 1..64
        self.assertEqual(body["stop"], ["END"])
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertNotIn("response_format", body)          # a BNF grammar is left out
        body = s._request_body(backend_pb2.PredictOptions(
            Messages=[backend_pb2.Message(role="user", content="hi")], Grammar=json.dumps({"type": "object"})))
        self.assertEqual(body["response_format"]["json_schema"]["schema"], {"type": "object"})


FAKE_SETUP = """#!/bin/sh
# Stands in for Strata's setup.sh: records its arguments, writes a run config.
echo "$@" > setup-args.txt
quant=""
while [ $# -gt 0 ]; do
  case "$1" in --model) quant="$2"; shift ;; esac
  shift
done
lower=$(echo "$quant" | tr '[:upper:]' '[:lower:]')
printf '{"exe": "x", "args": [], "cwd": "%s"}' "$(pwd)" > "strata-$lower.json"
"""


def _fake_strata_tarball(folder):
    """A tar.gz shaped like GitHub's source archives, holding FAKE_SETUP."""
    import io
    import tarfile
    path = os.path.join(folder, "strata.tar.gz")
    with tarfile.open(path, "w:gz") as tar:
        data = FAKE_SETUP.encode()
        info = tarfile.TarInfo("Strata-abc123/setup.sh")
        info.size, info.mode = len(data), 0o755
        tar.addfile(info, io.BytesIO(data))
    return path


class TestPrepare(unittest.TestCase):
    def test_detect(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
                         "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf", "README.md"):
                open(os.path.join(d, name), "w").close()
            self.assertEqual(len(prepare.gguf_shards(d)), 2)
            self.assertEqual(prepare.detect(d, {}), ("qwen", "IQ2_XS"))
            self.assertEqual(prepare.detect(d, {"family": "coder", "quant": "IQ1_M"}), ("coder", "IQ1_M"))
            with self.assertRaises(ValueError):
                prepare.detect(d, {"family": "llama"})
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"), "w").close()
            self.assertEqual(prepare.detect(d, {}), ("swift", "Q2_0"))
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"), "w").close()
            self.assertEqual(prepare.detect(d, {}), ("unsloth", "UD-IQ4_XS"))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                prepare.detect(d, {})

    def test_run_config_name(self):
        self.assertEqual(prepare.run_config_name("qwen", "IQ2_XS"), "strata-iq2_xs.json")
        self.assertEqual(prepare.run_config_name("unsloth", "UD-IQ4_XS"), "strata-unsloth-ud-iq4_xs.json")

    def test_prepare_runs_setup_once(self):
        with tempfile.TemporaryDirectory() as models:
            gguf = os.path.join(models, "strata", "models", "IQ2_XS")
            os.makedirs(gguf)
            for i in (1, 2):
                open(os.path.join(gguf, f"Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{i}-of-00002.gguf"), "w").close()
            old = prepare.STRATA_TARBALL
            prepare.STRATA_TARBALL = "file://" + _fake_strata_tarball(models) + "#{ref}"
            try:
                opts = {"family": "qwen", "quant": "IQ2_XS", "vram_reserve_mib": "1500"}
                cfg = prepare.prepare(gguf, opts, models, 65536)
            finally:
                prepare.STRATA_TARBALL = old
            strata = os.path.join(models, "strata", "Strata")
            self.assertEqual(cfg, os.path.join(strata, "strata-iq2_xs.json"))
            with open(os.path.join(strata, "setup-args.txt")) as f:
                args = f.read().split()
            for want in (["--gguf-dir", gguf], ["--context", "65536"], ["--vram-reserve-mib", "1500"],
                         ["--family", "qwen"], ["--model", "IQ2_XS"], ["--vision", "no"]):
                i = args.index(want[0])
                self.assertEqual(args[i:i + 2], want)
            self.assertIn("--no-start", args)
            with open(os.path.join(strata, ".strata-ref")) as f:
                self.assertEqual(f.read().strip(), prepare.STRATA_REF)
            # The second load finds the run config and runs nothing.
            os.remove(os.path.join(strata, "setup-args.txt"))
            self.assertEqual(prepare.prepare(gguf, opts, models, 65536), cfg)
            self.assertFalse(os.path.exists(os.path.join(strata, "setup-args.txt")))

    def test_failed_setup_is_reported(self):
        with tempfile.TemporaryDirectory() as models:
            strata = os.path.join(models, "strata", "Strata")
            os.makedirs(strata)
            with open(os.path.join(strata, "setup.sh"), "w") as f:
                f.write("#!/bin/sh\necho 'no GPU found' >&2\nexit 3\n")
            gguf = os.path.join(models, "strata", "models", "IQ2_XS")
            os.makedirs(gguf)
            open(os.path.join(gguf, "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"), "w").close()
            with self.assertRaises(RuntimeError) as cm:
                prepare.prepare(gguf, {}, models, 0)
            self.assertIn("exit 3", str(cm.exception))

    def test_unsafe_archive_rejected(self):
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.tar.gz")
            with tarfile.open(path, "w:gz") as tar:
                info = tarfile.TarInfo("Strata-x/../../escape.sh")
                info.size = 0
                tar.addfile(info, io.BytesIO(b""))
            old = prepare.STRATA_TARBALL
            prepare.STRATA_TARBALL = "file://" + path + "#{ref}"
            try:
                with self.assertRaises(ValueError):
                    prepare.ensure_source(os.path.join(d, "Strata"), "x")
            finally:
                prepare.STRATA_TARBALL = old
            self.assertFalse(os.path.exists(os.path.join(os.path.dirname(d), "escape.sh")))


class _FakeServer:
    """Stands in for StrataServer: replays SSE chunks as Strata sends them."""

    def __init__(self, chunks):
        self.chunks = chunks

    def alive(self):
        return True

    def stream_chat(self, body, is_active):
        yield from self.chunks


class _Ctx:
    def is_active(self):
        return True

    def abort(self, code, details):
        raise AssertionError(f"aborted: {code} {details}")


def _chunk(delta, usage=None):
    c = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    if usage:
        c["usage"] = usage
    return c


TOOL_STREAM = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"reasoning_content": "Need the "}),
    _chunk({"reasoning_content": "kitchen state."}),
    _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                            "function": {"name": "GetLiveContext", "arguments": ""}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"area\": "}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"Kitchen\"}"}}]}),
    _chunk({}, usage={"prompt_tokens": 5502, "completion_tokens": 33}),
]


class TestTranslation(unittest.TestCase):
    def _run(self, streaming, chunks=TOOL_STREAM):
        s = backend.BackendServicer()
        s.server = _FakeServer(chunks)
        req = backend_pb2.PredictOptions(Messages=[backend_pb2.Message(role="user", content="temp?")])
        return list(s._generate(req, _Ctx(), streaming=streaming))

    def test_tool_call_non_streaming(self):
        [reply] = self._run(streaming=False)
        d = reply.chat_deltas[0]
        self.assertEqual(d.reasoning_content, "Need the kitchen state.")
        self.assertEqual(d.content, "")
        self.assertEqual(len(d.tool_calls), 1)
        tc = d.tool_calls[0]
        self.assertEqual((tc.id, tc.name), ("call_1", "GetLiveContext"))
        self.assertEqual(json.loads(tc.arguments), {"area": "Kitchen"})
        self.assertEqual((reply.prompt_tokens, reply.tokens), (5502, 33))

    def test_tool_call_streaming(self):
        replies = self._run(streaming=True)
        deltas = [d for r in replies for d in r.chat_deltas]
        self.assertEqual("".join(d.reasoning_content for d in deltas), "Need the kitchen state.")
        calls = [tc for d in deltas for tc in d.tool_calls]
        self.assertEqual(calls[0].name, "GetLiveContext")
        self.assertEqual(json.loads("".join(tc.arguments for tc in calls)), {"area": "Kitchen"})
        self.assertEqual((replies[-1].prompt_tokens, replies[-1].tokens), (5502, 33))
        self.assertFalse(replies[-1].chat_deltas[0].content)   # the counts reply repeats nothing

    def test_plain_answer(self):
        chunks = [_chunk({"content": "It is "}), _chunk({"content": "21 degrees."}),
                  _chunk({}, usage={"prompt_tokens": 10, "completion_tokens": 4})]
        [reply] = self._run(streaming=False, chunks=chunks)
        self.assertEqual(reply.message, b"It is 21 degrees.")
        self.assertEqual(len(reply.chat_deltas[0].tool_calls), 0)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(STRATA_DIR and os.path.isfile(os.path.join(STRATA_DIR, "serve", "server.py")),
                     "set STRATA_DIR to a Strata checkout to run the end-to-end tests")
class TestWithMockEngine(unittest.TestCase):
    """The whole path: LocalAI's gRPC calls -> this backend -> Strata's server (mock engine)."""

    @classmethod
    def setUpClass(cls):
        cls.addr = f"127.0.0.1:{_free_port()}"
        cls.service = subprocess.Popen([sys.executable, "backend.py", "--addr", cls.addr])
        cls.channel = grpc.insecure_channel(cls.addr)
        grpc.channel_ready_future(cls.channel).result(timeout=30)
        cls.stub = backend_pb2_grpc.BackendStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.service.terminate()
        cls.service.wait(timeout=90)

    def _load(self):
        res = self.stub.LoadModel(backend_pb2.ModelOptions(
            Model=STRATA_DIR, Options=["engine:mock", "load_timeout:120"]))
        self.assertTrue(res.success, res.message)

    def _req(self, **kw):
        return backend_pb2.PredictOptions(
            Messages=[backend_pb2.Message(role="user", content="Say hello")], UseTokenizerTemplate=True,
            Tokens=64, **kw)

    def test_1_health(self):
        self.assertEqual(self.stub.Health(backend_pb2.HealthMessage()).message, b"OK")

    def test_2_predict_before_load(self):
        with self.assertRaises(grpc.RpcError) as cm:
            self.stub.Predict(self._req())
        self.assertEqual(cm.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_3_predict(self):
        self._load()
        reply = self.stub.Predict(self._req())
        self.assertTrue(reply.message, "empty answer")
        self.assertEqual(reply.chat_deltas[0].content, reply.message.decode())
        self.assertGreater(reply.prompt_tokens, 0)
        self.assertGreater(reply.tokens, 0)

    def test_4_stream(self):
        self._load()
        replies = list(self.stub.PredictStream(self._req(Metadata={"enable_thinking": "false"})))
        text = "".join(r.chat_deltas[0].content for r in replies if r.chat_deltas)
        self.assertTrue(text, "empty stream")
        self.assertEqual(text, b"".join(r.message for r in replies).decode())
        self.assertGreater(replies[-1].tokens, 0)           # the last reply carries the counts

    def test_5_free_stops_strata(self):
        self._load()
        self.assertTrue(self.stub.Free(backend_pb2.HealthMessage()).success)
        with self.assertRaises(grpc.RpcError) as cm:
            self.stub.Predict(self._req())
        self.assertEqual(cm.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_6_bad_option_fails_load(self):
        res = self.stub.LoadModel(backend_pb2.ModelOptions(Model=STRATA_DIR, Options=["vram_reserve_mib:lots"]))
        self.assertFalse(res.success)
        self.assertIn("vram_reserve_mib", res.message)


if __name__ == "__main__":
    unittest.main()
