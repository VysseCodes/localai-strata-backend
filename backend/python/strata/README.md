# Strata backend

Runs [Strata](https://github.com/Niko1221/Strata) (Qwen3.8-Flash-Next on one GPU
plus system RAM) as a LocalAI backend.

Strata ships its own OpenAI-compatible server that owns the chat template,
reasoning and tool-call parsing, and the engine process. This backend does not
duplicate any of that: `LoadModel` starts Strata's server as a private child
process (127.0.0.1, a free port, a random API key passed in the environment),
and `Predict` / `PredictStream` forward LocalAI's chat requests to it and
return content, reasoning and tool calls as `ChatDelta`s.

## Preparing the model

Strata's setup downloads the GGUFs (~70 GB) and builds a pack for the GPU. Run
it once, outside LocalAI, into the models folder:

```
./setup.sh --yes --family qwen --model IQ2_XS --context 65536 --vision no --no-start --no-browser
```

Point `parameters.model` at the run config it writes (`strata-<model>.json`),
or at the Strata folder if it holds only one.

## Model config

```yaml
name: strata
backend: strata
parameters:
  model: strata/Strata/strata-iq2_xs.json
template:
  use_tokenizer_template: true
options:
  - vram_reserve_mib:1500
```

| Option | Meaning |
| --- | --- |
| `vram_reserve_mib:N` | VRAM the engine leaves free. Strata sizes its expert cache from the VRAM free **when it starts**, so models already loaded are not counted; reserve only what others may still grow into. |
| `load_timeout:S` | Seconds to wait for the model to load (default 1800). |
| `reasoning_effort:E` | Thinking level for requests that set none: `none`, `low`, `medium`, `high`. |
| `engine:mock` | Strata's mock engine, for tests. |

`use_tokenizer_template: true` is required: Strata templates the conversation
itself, so it needs the messages rather than a prompt LocalAI rendered.

## Thinking

LocalAI turns a request's `reasoning_effort` into the `enable_thinking`
metadata flag; `enable_thinking=false` becomes Strata's `reasoning_effort:
"none"`.

## Notes

- Text only: images, video and audio are dropped (with a log line).
- Strata answers one request at a time unless its config sets `parallel`.
- The Strata server gets SIGTERM if this backend dies (`PR_SET_PDEATHSIG`),
  and stops its engine on SIGTERM, so a killed backend does not leave the GPU
  occupied.

## Tests

```
make test
```

Unit tests need nothing else. The end-to-end tests run Strata's server with its
mock engine (no GPU): set `STRATA_DIR` to a Strata checkout with a `.venv`.
