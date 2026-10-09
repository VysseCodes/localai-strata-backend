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

## 2. Install the model

**Through LocalAI (recommended).** One request: LocalAI downloads the ~68 GB of
model files with its own progress and resume, writes the model config, and
installs the `strata` backend if it is missing (from a backend gallery that
lists it, step 1):

```
curl -s http://localhost:8080/models/apply -H "Authorization: Bearer $LOCALAI_KEY" -H "Content-Type: application/json" -d '{"url":"github:VysseCodes/localai-strata-backend/gallery/strata-qwen3.8-flash-next-iq2_xs.yaml@main","name":"strata"}'
```

Watch it under Operate → Activity. To list the model in LocalAI's Models
catalog instead, add `https://raw.githubusercontent.com/VysseCodes/localai-strata-backend/main/gallery/index.yaml`
to `LOCALAI_GALLERIES` beside LocalAI's own gallery.

**The first load prepares the model for the machine,** once: the backend
fetches Strata's source (pinned to a tested commit) into `<models>/strata/Strata`
and runs Strata's own setup on the files LocalAI downloaded. Setup builds the
expert pack for this GPU and downloads what is left (the engine and the ~5 GB
MTP draft layer), often 10-20 minutes. Progress is in LocalAI's log, lines
tagged `[strata]`. The request that started it waits for it, so start it from
a client without a short timeout (curl has none). If it is interrupted, the
next load continues where it stopped. If LocalAI's busy watchdog is on, give
it a timeout longer than that.

**By hand (alternative).** `setup/strata-setup.sh` runs Strata's full setup
outside LocalAI and writes `<models>/strata.yaml` pointing at its run config.
Defaults: IQ2_XS, 64K context, text only, CUDA; override with `STRATA_MODEL`,
`STRATA_CONTEXT`, `STRATA_VRAM_RESERVE_MIB`.

## 3. Use it

The model config (the gallery install writes it; a hand setup points
`parameters.model` at the run config instead, see `models/strata.yaml`):

```yaml
name: strata
backend: strata
parameters:
  model: strata/models/IQ2_XS
context_size: 65536
template:
  use_tokenizer_template: true
options:
  - family:qwen
  - quant:IQ2_XS
  - vram_reserve_mib:1500
  - load_timeout:1800
```

| Option | Meaning |
| --- | --- |
| `vram_reserve_mib:N` | VRAM the engine leaves free. Strata sizes its expert cache from the VRAM free **when it starts**, so models already loaded are not counted: reserve only what others may still grow into. |
| `load_timeout:S` | Seconds to wait for the model to load (default 1800). |
| `reasoning_effort:E` | Thinking for requests that set none: `none`, `low`, `medium`, `high`. |
| `family:F`, `quant:Q` | The model in a GGUF folder (read from the file names when not set). |
| `strata_home:DIR` | Where Strata and its data go (default `<models>/strata`). |
| `strata_ref:REF` | The Strata commit or tag to prepare with (default: the tested one). |
| `prepare_timeout:S` | Seconds the one-time preparation may take (default 4 hours). |

`context_size` is used when the model is prepared; changing it later needs
Strata's setup run again (`setup.sh --setup --context N` in `<models>/strata/Strata`).

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
