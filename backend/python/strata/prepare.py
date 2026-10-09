"""First-load preparation of a Strata model installed from LocalAI's model gallery.

The gallery entry downloads the model's GGUF shards through LocalAI (with its
progress and resume) into <models>/strata/models/<QUANT>/ and writes a config
whose parameters.model is that folder. The rest of Strata's setup is specific
to the machine (the engine for the GPU, the expert pack, the MTP draft layer),
so it runs here, once, the first time the model is loaded:

  1. Strata's source, pinned to STRATA_REF, is unpacked to <models>/strata/Strata
  2. Strata's own setup runs with --gguf-dir pointing at the downloaded shards,
     so it downloads only what is left (~5 GB) and builds the pack
  3. the run config it writes (strata-<model>.json) is what later loads use

Every step is resumable: an interrupted run continues where it stopped.
"""
import fcntl
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

# The Strata release this backend is tested with. Override per model with the
# `strata_ref:` option (a commit or tag); change it here to move everyone.
STRATA_REF = "fb58e0dbc8399662c0e47c76578c6e878b14f6cf"
STRATA_TARBALL = "https://codeload.github.com/Niko1221/Strata/tar.gz/{ref}"

# Strata's model families and the prefix setup gives their run configs
# (setup.py FAMILIES[...]["tag"]): strata-<tag><quant>.json.
FAMILY_TAGS = {"qwen": "", "swift": "swift-", "coder": "coder-", "unsloth": "unsloth-"}

# Shard names (setup.py FAMILIES[...]["file"] and MODELS[...]["file"]).
SHARD_RE = re.compile(
    r"^(?P<prefix>Swift-)?Qwen3\.8-Flash-Next-(?:GSQ-RCO-)?(?P<quant>[A-Za-z0-9_-]+?)-0000\d-of-0000\d\.gguf$")

DEFAULT_PREPARE_TIMEOUT_S = 4 * 3600


def log(*args):
    print("[strata]", *args, file=sys.stderr, flush=True)


def gguf_shards(folder):
    """The Strata model shards in `folder`, sorted (empty if none)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(n for n in names if SHARD_RE.match(n))


def detect(folder, options):
    """(family, quant) for the shards in `folder`; options win over the names."""
    family = options.get("family")
    quant = options.get("quant")
    shards = gguf_shards(folder)
    if shards and (not family or not quant):
        m = SHARD_RE.match(shards[0])
        quant = quant or m.group("quant")
        if not family:
            family = "swift" if m.group("prefix") else ("unsloth" if quant.startswith("UD-") else "qwen")
    if not family or not quant:
        raise ValueError(f"cannot tell which Strata model {folder} holds: set the family: and quant: options")
    if family not in FAMILY_TAGS:
        raise ValueError(f"family must be one of {', '.join(FAMILY_TAGS)}, not {family!r}")
    return family, quant


def run_config_name(family, quant):
    return f"strata-{FAMILY_TAGS[family]}{quant.lower()}.json"


def _fetch_source(dest, ref):
    """Strata's source at `ref`, unpacked to `dest` (atomically: a half-done
    unpack never looks finished)."""
    url = STRATA_TARBALL.format(ref=ref)
    log(f"downloading Strata {ref[:12]} from {url}")
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    tmp = dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        top = members[0].name.split("/", 1)[0]
        for m in members:
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            if m.issym() or m.islnk() or m.isdev():
                continue                          # the source has none; never follow one from a download
            m.name = parts[1]
            if m.name.startswith("/") or ".." in m.name.split("/"):
                raise ValueError(f"unsafe path in the Strata archive: {m.name}")
            tar.extract(m, tmp)
        log(f"unpacked {top}")
    with open(os.path.join(tmp, ".strata-ref"), "w") as f:
        f.write(ref + "\n")
    os.replace(tmp, dest)


def ensure_source(strata_dir, ref):
    marker = os.path.join(strata_dir, ".strata-ref")
    if os.path.isfile(os.path.join(strata_dir, "setup.sh")):
        have = open(marker).read().strip() if os.path.isfile(marker) else "unknown"
        if have != ref and have != "unknown":
            log(f"Strata in {strata_dir} is {have[:12]}, the backend expects {ref[:12]}: keeping it "
                "(delete the folder to update)")
        return
    os.makedirs(os.path.dirname(strata_dir), exist_ok=True)
    _fetch_source(strata_dir, ref)


# Becomes the child after setting PR_SET_PDEATHSIG (see backend.py).
_PDEATHSIG_EXEC = (
    "import ctypes, os, signal, sys\n"
    "try:\n"
    "    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)\n"
    "except OSError:\n"
    "    pass\n"
    "if os.getppid() != int(os.environ.pop('STRATA_LOCALAI_PARENT')):\n"
    "    sys.exit(1)\n"
    "os.execv(sys.argv[1], sys.argv[1:])\n"
)


def run_setup(strata_dir, gguf_dir, family, quant, context, options, timeout):
    cmd = ["/bin/sh", os.path.join(strata_dir, "setup.sh"), "--yes",
           "--family", family, "--model", quant, "--gguf-dir", gguf_dir,
           "--vision", "no", "--no-start", "--no-browser"]
    if context:
        cmd += ["--context", str(context)]
    if options.get("vram_reserve_mib"):
        cmd += ["--vram-reserve-mib", options["vram_reserve_mib"]]
    if options.get("gpu_backend"):
        cmd += ["--backend", options["gpu_backend"]]
    env = dict(os.environ)
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"):
        env.pop(key, None)                       # setup makes its own .venv with the system Python
    env["STRATA_LOCALAI_PARENT"] = str(os.getpid())
    log("running Strata's setup:", " ".join(cmd))
    proc = subprocess.Popen([sys.executable, "-c", _PDEATHSIG_EXEC] + cmd, cwd=strata_dir, env=env,
                            stdin=subprocess.DEVNULL, stdout=sys.stderr.fileno(), stderr=sys.stderr.fileno(),
                            start_new_session=True)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=60)
        raise TimeoutError(f"Strata's setup did not finish within {timeout:.0f} s; load again to continue it")
    if code != 0:
        raise RuntimeError(f"Strata's setup failed (exit {code}); its output is in the log above. "
                           "Load the model again to retry: it continues where it stopped")


def prepare(gguf_dir, options, models_path, context):
    """The run config for the model whose shards are in `gguf_dir`, running
    Strata's setup first if this machine has not prepared it yet."""
    gguf_dir = os.path.abspath(gguf_dir)
    family, quant = detect(gguf_dir, options)
    home = options.get("strata_home") or "strata"
    if not os.path.isabs(home):
        home = os.path.join(models_path or os.path.dirname(os.path.dirname(gguf_dir)), home)
    strata_dir = os.path.join(home, "Strata")
    config = os.path.join(strata_dir, run_config_name(family, quant))
    if os.path.isfile(config):
        return config

    os.makedirs(home, exist_ok=True)
    with open(os.path.join(home, ".prepare.lock"), "w") as lock:
        log(f"first load of {family} {quant}: preparing it for this machine (one time; ~5 GB to download "
            "and the expert pack to build, often 10-20 minutes)")
        fcntl.flock(lock, fcntl.LOCK_EX)         # another load preparing the same model: wait for it
        if os.path.isfile(config):
            return config
        started = time.monotonic()
        ensure_source(strata_dir, options.get("strata_ref") or STRATA_REF)
        timeout = float(options.get("prepare_timeout") or DEFAULT_PREPARE_TIMEOUT_S)
        run_setup(strata_dir, gguf_dir, family, quant, context, options, timeout)
        if not os.path.isfile(config):
            raise RuntimeError(f"Strata's setup finished but did not write {config}")
        log(f"prepared in {(time.monotonic() - started) / 60:.1f} min: {config}")
        return config
