# localai-strata-backend

A [LocalAI](https://github.com/mudler/LocalAI) backend for
[Strata](https://github.com/Niko1221/Strata), which runs Qwen3.8-Flash-Next on
one GPU plus system RAM. Install it from LocalAI's backend catalog like any
other backend.

Strata ships its own OpenAI-compatible server that owns the chat template,
reasoning and tool-call parsing, and the engine. This backend does not
duplicate any of that: `LoadModel` starts Strata's server as a private child
process (127.0.0.1, a free port, a random API key passed in the environment),
and LocalAI's chat requests are forwarded to it, with content, reasoning and
tool calls returned the way LocalAI expects. The backend image is small and
CPU-only; the model and its GPU engine come from Strata's own setup.

## 1. Install the backend

Either way gives you a backend named `strata`.

**From the image (quickest).** In LocalAI: Operate → Runtime → Backends →
**From URL**, image `ghcr.io/vyssecodes/localai-strata-backend:latest`, name
`strata`.

**From the catalog (shows up next to LocalAI's own backends, upgradable).**
Add this repo's gallery beside LocalAI's in the service environment, then
restart LocalAI and install `strata` from the Catalog tab:

```
LOCALAI_BACKEND_GALLERIES='[{"name":"localai","url":"https://index.localai.io/backends","mirrors":["github:mudler/LocalAI/backend/index.yaml@master","oci://quay.io/go-skynet/local-ai-backends:gallery-backends"],"artifact_verification":{"issuer":"https://token.actions.githubusercontent.com","identity":"https://github.com/mudler/LocalAI/.github/workflows/gallery_publish.yml@refs/heads/master"}},{"name":"strata","url":"https://raw.githubusercontent.com/VysseCodes/localai-strata-backend/main/index.yaml"}]'
```

The first entry is LocalAI's default gallery; setting the variable replaces
the default, so keep it in the list.

## 2. Prepare the model (once)

On the LocalAI machine, as root. It downloads ~70 GB and builds a pack for the
GPU, so it can take an hour or more; run it again to resume:

```
bash setup/strata-setup.sh
```

It puts Strata under `<models>/strata/` and writes `<models>/strata.yaml`.
Defaults: Qwen3.8-Flash-Next IQ2_XS, 64K context, text only, CUDA; override
with `STRATA_MODEL`, `STRATA_CONTEXT`, `STRATA_VRAM_RESERVE_MIB`.

## 3. Use it

`strata.yaml` (written by the setup script; reference copy in `models/`):

```yaml
name: strata
backend: strata
parameters:
  model: strata/Strata/strata-iq2_xs.json
template:
  use_tokenizer_template: true
options:
  - vram_reserve_mib:1500
  - load_timeout:1800
```

| Option | Meaning |
| --- | --- |
| `vram_reserve_mib:N` | VRAM the engine leaves free. Strata sizes its expert cache from the VRAM free **when it starts**, so models already loaded are not counted: reserve only what others may still grow into. |
| `load_timeout:S` | Seconds to wait for the model to load (default 1800). |
| `reasoning_effort:E` | Thinking for requests that set none: `none`, `low`, `medium`, `high`. |

`use_tokenizer_template: true` is required: Strata templates the conversation
itself. A request's `reasoning_effort: "none"` turns thinking off.

### Sharing one GPU with another LLM

`models/main-llm.yaml` is a LocalAI failover chain: clients call `main-llm`,
another model answers normally, and pinning the chain to `strata` swaps Strata
in (`POST /api/failover/main-llm/pin {"target":"strata"}`, `DELETE` to go
back). Give both models the same `concurrency_groups` entry so loading one
evicts the other, and don't mark either `warm`.

## Building

GitHub Actions (`.github/workflows/build.yml`) runs the tests, including an
end-to-end run against Strata's server with its mock engine (no GPU), then
builds the image with LocalAI's own `backend/Dockerfile.python` at the LocalAI
release in `LOCALAI_REF` and pushes it to GHCR. Keep `LOCALAI_REF` at the
LocalAI version you run.

Locally, inside a LocalAI checkout:

```
cp -r backend/python/strata <LocalAI>/backend/python/
```

```
docker build -f backend/Dockerfile.python --build-arg BACKEND=strata -t localai-strata-backend .
```

## License

MIT. Strata itself is MIT-licensed by its authors and is downloaded by the
setup script, not included here.
