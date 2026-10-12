"""The console's model: which provider (Claude or GPT), which model, and the
API key — one choice that both the analyst and the LLM planner use.

The key goes into OVERLORD's own key store ($OVERLORD_HOME/keys.json, mode
0600, `{provider: key}`), the file the daemon reads on every model call, so a
key set here takes effect on the next call with no restart. It is never
written anywhere else, never logged and never sent back: reads report only
whether a key is set and where it comes from. An environment variable
(ANTHROPIC_API_KEY / OPENAI_API_KEY) wins over the store, as it does in
OVERLORD, and the UI says so.

The provider/model choice lives next to it in $OVERLORD_HOME/console-model.json.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

PROVIDERS: dict[str, dict[str, str]] = {
    "anthropic": {"label": "Anthropic (Claude)", "key_env": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-5"},
    "openai": {"label": "OpenAI (GPT)", "key_env": "OPENAI_API_KEY", "default_model": "gpt-5.5"},
}
DEFAULT_PROVIDER = "anthropic"
# OpenAI's /v1/models lists embeddings, speech, images… — none of them can plan or chat
_NOT_CHAT = ("embedding", "tts", "whisper", "dall-e", "audio", "realtime", "image", "moderation",
             "transcribe", "search", "davinci", "babbage", "computer-use")


class SettingsError(ValueError):
    """A setting the console refuses (400)."""


def _home() -> Path:
    return Path(os.environ.get("OVERLORD_HOME") or Path.home() / ".overlord")


class ModelSettings:
    def __init__(self, home: Optional[Path] = None, env: Optional[dict] = None):
        self.home = Path(home) if home else _home()
        self.env = os.environ if env is None else env
        self._lock = threading.Lock()

    # ------------------------------------------------------------ files

    @property
    def keys_file(self) -> Path:
        return self.home / "keys.json"

    @property
    def choice_file(self) -> Path:
        return self.home / "console-model.json"

    def _read_keys(self) -> dict:
        try:
            return json.loads(self.keys_file.read_text()) or {}
        except (OSError, ValueError):
            return {}

    def _write_keys(self, keys: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.keys_file.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(keys, f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.keys_file)

    def _choice(self) -> dict:
        try:
            c = json.loads(self.choice_file.read_text())
        except (OSError, ValueError):
            c = {}
        provider = c.get("provider") or self.env.get("CONSOLE_LLM_PROVIDER") or DEFAULT_PROVIDER
        if provider not in PROVIDERS:
            provider = DEFAULT_PROVIDER
        model = c.get("model") if "model" in c else (self.env.get("CONSOLE_LLM_MODEL") or "")
        return {"provider": provider, "model": model or ""}

    # ------------------------------------------------------------ reads

    def key_source(self, provider: str) -> str:
        """"env", "store" or "none" — never the key."""
        if self.env.get(PROVIDERS[provider]["key_env"]):
            return "env"
        return "store" if self._read_keys().get(provider) else "none"

    def current(self) -> dict:
        """What a model call uses: {provider, model} with the default filled in."""
        c = self._choice()
        return {"provider": c["provider"], "model": c["model"] or PROVIDERS[c["provider"]]["default_model"]}

    def public(self) -> dict:
        c = self._choice()
        cur = self.current()
        return {
            "provider": c["provider"],
            "model": c["model"],                              # "" = the provider's default
            "effective_model": cur["model"],
            "ready": self.key_source(c["provider"]) != "none",
            "providers": [{"id": p, "label": v["label"], "key_env": v["key_env"],
                           "default_model": v["default_model"], "key": self.key_source(p)}
                          for p, v in PROVIDERS.items()],
        }

    # ------------------------------------------------------------ writes

    def update(self, provider: Optional[str] = None, model: Optional[str] = None,
               key: Optional[str] = None, clear_key: bool = False) -> dict:
        """Change the provider/model and/or set or clear that provider's key.
        `key` None or "" leaves the stored key as it is."""
        with self._lock:
            c = self._choice()
            p = provider or c["provider"]
            if p not in PROVIDERS:
                raise SettingsError(f"provider must be one of {', '.join(PROVIDERS)}")
            m = c["model"] if model is None else str(model).strip()
            if p != c["provider"] and model is None:
                m = ""                                        # a model id does not carry across providers
            if len(m) > 200 or any(ch.isspace() for ch in m):
                raise SettingsError("model must be a model id (no spaces)")
            if key is not None and str(key).strip():
                k = str(key).strip()
                if len(k) > 4096 or any(ch.isspace() for ch in k):
                    raise SettingsError("that does not look like an API key")
                keys = self._read_keys()
                keys[p] = k
                self._write_keys(keys)
            elif clear_key:
                keys = self._read_keys()
                if keys.pop(p, None) is not None:
                    self._write_keys(keys)
            self.home.mkdir(parents=True, exist_ok=True)
            self.choice_file.write_text(json.dumps({"provider": p, "model": m}))
        return self.public()

    # ------------------------------------------------------------ provider calls

    def models(self, list_fn: Callable[..., list], provider: Optional[str] = None) -> list[dict]:
        """The provider's live chat models, through OVERLORD (which holds the key)."""
        p = provider or self._choice()["provider"]
        if p not in PROVIDERS:
            raise SettingsError(f"provider must be one of {', '.join(PROVIDERS)}")
        rows = list_fn(provider=p) or []
        out = []
        for r in rows:
            mid = str(r.get("id") if isinstance(r, dict) else r)
            if p == "openai" and any(t in mid.lower() for t in _NOT_CHAT):
                continue
            out.append({"id": mid, "name": (r.get("name") if isinstance(r, dict) else None) or mid})
        return out

    def complete_kwargs(self) -> dict[str, Any]:
        cur = self.current()
        return {"provider": cur["provider"], "model": cur["model"]}
