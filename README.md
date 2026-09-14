This was initially started as a quick project to train steganographic models for workshops for ICML 2026.  The `ciphers/variable_naming_in_python_v1` folder has this. I did not have time to finish, so instead I re-started the project for Neel Nanda's MATS stream application in September 2026. The work for Neel is in `ciphers/kirchenbauer_et_al`

This is a work in progress.

## Codex from Python

The [official Codex Python SDK](https://learn.chatgpt.com/docs/codex-sdk) is included
in `requirements.txt`. To install just this dependency in the project environment:

```bash
conda run -n stego python -m pip install 'openai-codex==0.154.0'
```

Run Python in the `stego` environment from the repository root:

```python
from openai_codex import Codex

with Codex() as codex:
    thread = codex.thread_start()
    result = thread.run("hello")
    print(result.final_response)
```

The package installs its matching Codex CLI runtime automatically. It reuses your
existing Codex login when run as the same user with the same `CODEX_HOME`
(the default is `~/.codex`), so an existing ChatGPT login needs no API key. If you
have not logged in on this machine, run `codex login` first. See the
[authentication documentation](https://learn.chatgpt.com/docs/auth).

Call `thread.run(...)` again to continue the conversation, or
`codex.thread_start()` to create another agent thread. The SDK also provides
`AsyncCodex` for asynchronous workflows. Threads use the configured sandbox
unless you pass a `Sandbox` preset explicitly.

TODO(hadriano) improve AGENTS.md with some of the stuff from https://github.com/4gatepylon/SAEScoping/blob/main/AGENTS.md (via trial and error).
