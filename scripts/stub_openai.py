#!/usr/bin/env python3
"""A tiny OpenAI-compatible endpoint that answers NuggetRecall's two prompts.

This exists for one reason: to make the container's own test run actually call an
LLM.

The code submission's pre-upload test used to point OPENAI_BASE_URL at a closed
port. NuggetRecall then correctly fell back, wrote an all-zero leaderboard, and
every TIRA check passed, because the format checker only looks at the shape of the
file and never asks whether a measure varies. The deterministic-cache proof passed
too, since both the first run and the replay did nothing and so agreed exactly.
The result was a green rehearsal that proved the image builds and boots, and
proved nothing at all about the path that does the judging.

With this stub reachable and --allow-network, the container really calls out, the
prompt cache is really written, and the replay really replays. Whatever the stub
says is not the point and is never scored. What is being tested is that the judge
can reach an endpoint from inside the image, that minima-llm works there, that
CACHE_DIR is writable, and that the emitted leaderboard separates systems.

Replies are deterministic and depend only on the prompt, so the cache behaves as
it would against a real server.

    python scripts/stub_openai.py --port 8000 --host 0.0.0.0

Bind 0.0.0.0 and point the container at the docker bridge gateway (172.17.0.1 on
a standard Linux runner), because 127.0.0.1 inside a container is the container.
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Deliberately varied labels. An all-"support" stub would make every system score
# identically, the leaderboard would be constant, and check_upload_dir would
# refuse it, which would look like a failure of the judge rather than of the stub.
_LABELS = ("support", "partial", "no")


def _nuggets(n: int = 6) -> str:
    return "\n".join(
        f"{i}. [{'VITAL' if i <= 3 else 'OKAY'}] Stub fact number {i} for this request."
        for i in range(1, n + 1))


def _judgments(prompt: str) -> str:
    """Label every numbered fact the prompt lists, varying by the answer's text.

    The label depends on a hash of the response text, so different systems get
    different scores and the leaderboard has something to rank.
    """
    facts = re.findall(r"^\s*(\d+)\.", prompt, flags=re.MULTILINE)
    count = max((int(f) for f in facts), default=6)
    m = re.search(r"RESPONSE TO JUDGE:\n(.*?)\n\nFACTS TO CHECK", prompt, flags=re.S)
    seed = sum(ord(c) for c in (m.group(1) if m else prompt)[:400])
    return "\n".join(f"{i}: {_LABELS[(seed + i) % len(_LABELS)]}"
                     for i in range(1, count + 1))


def _reply_for(prompt: str) -> str:
    if "FACTS TO CHECK" in prompt:
        return _judgments(prompt)
    return _nuggets()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send({"error": "bad json"}, 400)
            return
        msgs = req.get("messages") or []
        prompt = "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
        text = _reply_for(prompt)
        self._send({
            "id": "stub", "object": "chat.completion", "model": req.get("model", MODEL_ID),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    def log_message(self, *a):
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    srv = HTTPServer((a.host, a.port), Handler)
    print(f"stub OpenAI endpoint on {a.host}:{a.port}, model {MODEL_ID}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
