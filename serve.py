#!/usr/bin/env python3
import json
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from search import TFIDFSearch

TEMPLATE = (Path(__file__).parent / "templates" / "index.html").read_text()
ENGINE = None
CURRENT_FOLDER = None
CURRENT_MODE = None
CURRENT_K1 = 1.5
CURRENT_B = 0.75


def get_engine(folder, mode, k1, b):
    global ENGINE, CURRENT_FOLDER, CURRENT_MODE, CURRENT_K1, CURRENT_B
    if (ENGINE is None or folder != CURRENT_FOLDER or mode != CURRENT_MODE
            or k1 != CURRENT_K1 or b != CURRENT_B):
        ENGINE = TFIDFSearch(folder, mode=mode, k1=k1, b=b)
        CURRENT_FOLDER = folder
        CURRENT_MODE = mode
        CURRENT_K1 = k1
        CURRENT_B = b
    return ENGINE


class ReusableServer(HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args, file=sys.stderr)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(TEMPLATE.encode()))
            self.end_headers()
            self.wfile.write(TEMPLATE.encode())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        global ENGINE, CURRENT_FOLDER, CURRENT_MODE
        try:
            if self.path == "/api/search":
                data = self._read_json()
                query = data.get("query", "").strip()
                folder = data.get("folder", CURRENT_FOLDER or ".")
                mode = data.get("mode", "tfidf")
                k1 = float(data.get("k1", 1.5))
                b = float(data.get("b", 0.75))
                top_k = int(data.get("top_k", 10))

                if not query:
                    self._send_json({"error": "query required"}, 400)
                    return

                engine = get_engine(folder, mode, k1, b)
                results = engine.search(query, top_k=top_k)
                self._send_json({"results": [{"file": r[0], "score": round(r[1], 4)}
                                             for r in results],
                                 "total_docs": len(engine.docs),
                                 "vocab_size": len(engine.vocab),
                                 "mode": mode})

            elif self.path == "/api/reindex":
                ENGINE = None
                CURRENT_FOLDER = None
                CURRENT_MODE = None
                self._send_json({"status": "ok"})

            else:
                self._send_json({"error": "not found"}, 404)

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self._send_json({"error": str(e)}, 500)


def ensure_nltk_data():
    import nltk
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")

        except LookupError:
            nltk.download(pkg, quiet=True)


def main():
    global CURRENT_FOLDER, CURRENT_MODE
    ensure_nltk_data()

    import argparse
    parser = argparse.ArgumentParser(description="Markdown search engine — web UI")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Path to folder containing .md files")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    if args.folder:
        CURRENT_FOLDER = args.folder
        CURRENT_MODE = "tfidf"

    server = ReusableServer(("0.0.0.0", args.port), Handler)
    print(f"http://localhost:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
