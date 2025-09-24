#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "sample_data")
PAPERS_PATH = os.path.join(DATA_DIR, "papers.json")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus_analysis.json")

PAPERS = []
PAPER_BY_ID = {}
CORPUS = {}

def log_line(method, path, status, note=""):
    msg = {200:"OK",400:"Bad Request",404:"Not Found",500:"Error"}.get(status,"")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tail = f" ({note})" if note else ""
    print(f"[{now}] {method} {path} - {status} {msg}{tail}")
    sys.stdout.flush()

def safe_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_data():
    global PAPERS, PAPER_BY_ID, CORPUS
    PAPERS = safe_load(PAPERS_PATH)
    PAPER_BY_ID = {str(p.get("arxiv_id")): p for p in PAPERS if p.get("arxiv_id")}
    CORPUS = safe_load(CORPUS_PATH)

def tokenize(text):
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]

def count_occurrences(text, term):
    term_esc = re.escape(term)
    try:
        pat = re.compile(rf"\b{term_esc}\b", re.IGNORECASE)
    except re.error:
        pat = re.compile(re.escape(term), re.IGNORECASE)
    return len(pat.findall(text or ""))

class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_err(self, status, msg):
        self.send_json({"error": msg}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            # GET /papers
            if path == "/papers":
                items = [{
                    "arxiv_id": p.get("arxiv_id"),
                    "title": p.get("title"),
                    "authors": p.get("authors", []),
                    "categories": p.get("categories", [])
                } for p in PAPERS]
                self.send_json(items)
                log_line("GET", path, 200, f"{len(items)} results")
                return

            # GET /papers/{id}
            if path.startswith("/papers/"):
                aid = unquote(path[len("/papers/"):])
                paper = PAPER_BY_ID.get(aid)
                if not paper:
                    self.send_err(404, f"Unknown paper id: {aid}")
                    log_line("GET", path, 404)
                    return
                detail = {
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("abstract", ""),
                    "categories": paper.get("categories", []),
                    "published": paper.get("published") or "",
                    "abstract_stats": paper.get("abstract_stats") or {
                        "total_words": len(tokenize(paper.get("abstract",""))),
                        "unique_words": len(set(tokenize(paper.get("abstract","")))),
                        "total_sentences": max(1, len([s for s in re.split(r"[.!?]+", (paper.get("abstract","")).strip()) if s]))
                    }
                }
                self.send_json(detail)
                log_line("GET", path, 200)
                return

            # GET /search
            if path == "/search":
                q = (qs.get("q", [""])[0]).strip()
                if not q:
                    self.send_err(400, "Malformed search: missing or empty 'q'")
                    log_line("GET", f"{path}?{parsed.query}", 400)
                    return
                terms = [t for t in re.split(r"\s+", q) if t]
                results = []
                for p in PAPERS:
                    title = p.get("title","")
                    abstract = p.get("abstract","")
                    score, seen, ok = 0, set(), True
                    for t in terms:
                        ct = count_occurrences(title, t)
                        ca = count_occurrences(abstract, t)
                        if ct+ca == 0:
                            ok = False
                            break
                        score += ct+ca
                        if ct: seen.add("title")
                        if ca: seen.add("abstract")
                    if ok:
                        results.append({
                            "arxiv_id": p.get("arxiv_id"),
                            "title": title,
                            "match_score": score,
                            "matches_in": sorted(seen)
                        })
                self.send_json({"query": q, "results": results})
                log_line("GET", f"{path}?{parsed.query}", 200, f"{len(results)} results")
                return

            # GET /stats
            if path == "/stats":
                self.send_json(CORPUS)
                log_line("GET", path, 200)
                return

            # Unknown
            self.send_err(404, "Endpoint not found")
            log_line("GET", path, 404)

        except Exception as e:
            self.send_err(500, f"Server error: {type(e).__name__}: {e}")
            log_line("GET", path, 500)

def main():
    try:
        load_data()
    except Exception as e:
        print("Error loading data:", e, file=sys.stderr)
        sys.exit(1)

    port = 8080
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
