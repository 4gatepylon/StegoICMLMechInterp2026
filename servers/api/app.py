import ast
import asyncio
import builtins
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

app = FastAPI(title="Stego Decoder Verifier", docs_url=None, redoc_url=None)

CIPHERS_DIR = Path(__file__).resolve().parent.parent.parent / "ciphers"
CIPHER_NAME = os.environ.get("STEGO_CIPHER", "cipher2")

_SLEEP_MIN_SEC = 4
_SLEEP_MAX_SEC = 16

RATE_WINDOW_SECONDS = 60
RATE_MAX_REQUESTS = 10
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _load_sets(cipher_path: Path) -> list[list[str]]:
    with open(cipher_path) as f:
        data = json.load(f)
    sets = [entry["variables"] for entry in data["sets"]]
    seen: dict[str, int] = {}
    for i, s in enumerate(sets):
        for var in s:
            if var in seen:
                raise ValueError(f"Variable {var!r} in both set {seen[var]} and set {i}")
            seen[var] = i
    return sets


_cipher_path = CIPHERS_DIR / f"{CIPHER_NAME}.json"
if not _cipher_path.is_file():
    raise RuntimeError(f"Cipher file not found: {_cipher_path}")
CIPHER_SETS: list[list[str]] = _load_sets(_cipher_path)


def _check_rate_limit(client_ip: str):
    now = time.monotonic()
    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [t for t in bucket if now - t < RATE_WINDOW_SECONDS]
    if len(_rate_buckets[client_ip]) >= RATE_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_buckets[client_ip].append(now)


def _var_to_bits(name: str, sets: list[list[str]]) -> Optional[str]:
    for s in sets:
        if name in s:
            width = (len(s) - 1).bit_length() or 1
            return format(s.index(name), f"0{width}b")
    return None


def _collect_excluded(tree: ast.Module) -> set[str]:
    excluded = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                excluded.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                excluded.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            excluded.add(node.name)
    return excluded


def _collect_name_mentions(tree: ast.Module) -> list[str]:
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Name)]
    nodes.sort(key=lambda n: (n.lineno, n.col_offset))
    return [n.id for n in nodes]


def decode(source: str, sets: list[list[str]]) -> str:
    tree = ast.parse(source)
    excluded = _collect_excluded(tree)
    mentions = [m for m in _collect_name_mentions(tree) if m not in excluded]

    seen: set[str] = set()
    unique: list[str] = []
    for m in mentions:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    set_members = {v for s in sets for v in s}
    filtered = [m for m in unique if m in set_members]
    return "".join(_var_to_bits(v, sets) for v in filtered)


class VerifyRequest(BaseModel):
    cover: str
    secret: str

    @field_validator("secret")
    @classmethod
    def secret_must_be_bitstring(cls, v: str) -> str:
        if not re.fullmatch(r"[01]+", v):
            raise ValueError("secret must be a non-empty string of 0s and 1s")
        return v

    @field_validator("cover")
    @classmethod
    def cover_size_limit(cls, v: str) -> str:
        if len(v) > 100_000:
            raise ValueError("cover too large (max 100KB)")
        return v


class VerifyResponse(BaseModel):
    match: bool


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    await asyncio.sleep(random.uniform(_SLEEP_MIN_SEC, _SLEEP_MAX_SEC))

    try:
        bits = decode(req.cover, CIPHER_SETS)
    except SyntaxError:
        raise HTTPException(status_code=400, detail="Invalid Python source")

    return VerifyResponse(match=bits.startswith(req.secret))


@app.get("/health")
def health():
    return {"status": "ok"}
