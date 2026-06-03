"""
tr_server.py — A simple HTTP server hosting the tokenizer demo page
plus a JSON API. Stdlib only; no Flask/FastAPI.

Run:
    python tr_server.py             # default port 8000
    python tr_server.py --port 8080

Endpoints:
    GET  /                          → the demo HTML page
    GET  /api/tokenize?word=X       → JSON analysis of one word
    POST /api/tokenize_text         → body { "text": "..." } → sentence JSON
    GET  /static/<file>             → static assets (the CSS/JS bundled
                                       inline in the HTML, so this is
                                       only used if you split them out)

The server loads the tokenizer once at startup and reuses it across
requests. Single-process; concurrent requests are serialized by the
default ThreadingHTTPServer but the parser is thread-safe (no mutable
state across calls).
"""

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from tr_api import Tokenizer, TokenizerConfig


HERE = Path(__file__).parent

_TRUE_STRINGS = {"1", "true", "yes", "on", "t"}
_FALSE_STRINGS = {"0", "false", "no", "off", "f", ""}


def _to_bool(value):
    """Coerce a query/body value to True/False, or None if absent/unknown.
    None means 'let the tokenizer use its configured default'."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _TRUE_STRINGS:
        return True
    if s in _FALSE_STRINGS:
        return False
    return None


# The HTML page lives in a separate file so it can be edited without
# touching Python. Loaded once at server start.
def _load_html() -> str:
    path = HERE / "tr_demo.html"
    if not path.exists():
        raise FileNotFoundError(
            f"tr_demo.html not found next to tr_server.py at {path}. "
            f"The HTML page must be deployed alongside the server."
        )
    return path.read_text(encoding="utf-8")


class TokenizerHandler(BaseHTTPRequestHandler):
    """HTTP request handler. The `tokenizer` and `html` class attributes
    are set on the handler class (not instance) before binding the
    server, so they're shared across all request threads."""

    tokenizer: Tokenizer = None
    html: str = ""
    max_text_chars: int = 50000   # cap on /api/tokenize_text input (configurable)

    # -------- shared helpers --------

    def _send_json(self, payload, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, msg: str, status: int = 400):
        body = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Keep the access log tidy; suppress the default verbose format.
        sys.stderr.write(f"[server] {self.address_string()} {fmt % args}\n")

    # -------- routes --------

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._send_html(self.html)
            return
        if parsed.path == "/api/tokenize":
            self._handle_tokenize_get(parsed)
            return
        self._send_text(f"Not found: {parsed.path}", status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/tokenize_text":
            self._handle_tokenize_text_post()
            return
        self._send_text(f"Not found: {parsed.path}", status=404)

    # -------- handlers --------

    def _handle_tokenize_get(self, parsed):
        try:
            qs = parse_qs(parsed.query)
            words = qs.get("word", [])
            if not words:
                self._send_json({"error": "missing 'word' parameter"},
                                status=400)
                return

            # Per-request toggles (absent -> tokenizer default). Short query
            # names: suggest, tail (suffix-tail repair), alts (alternatives).
            def flag(name):
                vals = qs.get(name)
                return _to_bool(vals[0]) if vals else None

            result = self.tokenizer.tokenize(
                words[0],
                suggest=flag("suggest"),
                tail_repair=flag("tail"),
                alternatives=flag("alts"),
            )
            self._send_json(result)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def _handle_tokenize_text_post(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._send_json({"error": f"invalid JSON body: {e}"},
                                status=400)
                return
            text = (body.get("text") or "").strip()
            if not text:
                self._send_json({"error": "missing 'text' field"},
                                status=400)
                return
            # Cap on input size (configurable via --max-text-chars). Large
            # documents should be sent in chunks rather than one huge body.
            if len(text) > self.max_text_chars:
                self._send_json(
                    {"error": f"text too long (max {self.max_text_chars} chars)"},
                    status=400)
                return
            # Per-request toggles, e.g. {"options": {"tail_repair": false,
            # "suggest": false, "split_clitics": true, "alternatives": false}}.
            # Turning the lengthy ones off is the fast path for bulk/document
            # tokenization.
            opts = body.get("options") or {}
            result = self.tokenizer.tokenize_text(
                text,
                split_clitics=_to_bool(opts.get("split_clitics")),
                suggest=_to_bool(opts.get("suggest")),
                tail_repair=_to_bool(opts.get("tail_repair")),
                alternatives=_to_bool(opts.get("alternatives")),
            )
            self._send_json(result)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000,
                    help="port to bind (default: 8000)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="host to bind (default: 127.0.0.1 — "
                         "use 0.0.0.0 to accept external traffic)")
    ap.add_argument("--lexicon", default=str(HERE / "lexicon.json"),
                    help="lexicon JSON to load")
    ap.add_argument("--max-text-chars", type=int, default=50000,
                    help="max characters accepted by /api/tokenize_text "
                         "(default: 50000; raise for larger documents)")
    args = ap.parse_args()

    print(f"Loading tokenizer (lexicon: {args.lexicon}) ...", file=sys.stderr)
    cfg = TokenizerConfig(lexicon_path=Path(args.lexicon))
    TokenizerHandler.tokenizer = Tokenizer(cfg)
    TokenizerHandler.html = _load_html()
    TokenizerHandler.max_text_chars = args.max_text_chars
    print(f"Loaded. Serving on http://{args.host}:{args.port}/",
          file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), TokenizerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.server_close()


if __name__ == "__main__":
    main()
