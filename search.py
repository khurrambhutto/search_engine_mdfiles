#!/usr/bin/env python3
import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


class TFIDFSearch:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.docs = {}          # doc_name -> cleaned text
        self.vocab = set()
        self.tf = {}            # doc_name -> {word: tf}
        self.idf = {}           # word -> idf
        self.tfidf = {}         # doc_name -> {word: tfidf}
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _load_docs(self):
        for md_file in self.folder.rglob("*.md"):
            name = str(md_file.relative_to(self.folder))
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            tokens = self._tokenize(text)
            self.docs[name] = tokens
            self.vocab.update(tokens)

    def _compute_tf(self):
        for doc, tokens in self.docs.items():
            if not tokens:
                self.tf[doc] = {}
                continue
            total = len(tokens)
            counts = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            self.tf[doc] = {w: c / total for w, c in counts.items()}

    def _compute_idf(self):
        n = len(self.docs)
        for word in self.vocab:
            df = sum(1 for tokens in self.docs.values() if word in tokens)
            self.idf[word] = math.log(n / (1 + df)) + 1  # smooth idf

    def _compute_tfidf(self):
        for doc in self.docs:
            self.tfidf[doc] = {}
            for word, tf_val in self.tf.get(doc, {}).items():
                self.tfidf[doc][word] = tf_val * self.idf[word]

    def _build_index(self):
        self._load_docs()
        self._compute_tf()
        self._compute_idf()
        self._compute_tfidf()
        print(f"Indexed {len(self.docs)} documents, {len(self.vocab)} unique terms.", file=sys.stderr)

    def _query_tfidf(self, query: str) -> dict[str, float]:
        qtokens = self._tokenize(query)
        if not qtokens:
            return {}
        counts = defaultdict(int)
        for t in qtokens:
            counts[t] += 1
        qvec = {w: (c / len(qtokens)) * self.idf.get(w, 0) for w, c in counts.items()}

        scores = {}
        for doc, dvec in self.tfidf.items():
            dot = sum(qvec.get(w, 0) * dvec.get(w, 0) for w in qvec)
            norm_q = math.sqrt(sum(v ** 2 for v in qvec.values()))
            norm_d = math.sqrt(sum(v ** 2 for v in dvec.values()))
            if norm_q > 0 and norm_d > 0:
                scores[doc] = dot / (norm_q * norm_d)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores = self._query_tfidf(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def reload(self):
        self.docs.clear()
        self.vocab.clear()
        self.tf.clear()
        self.idf.clear()
        self.tfidf.clear()
        self._build_index()


def main():
    parser = argparse.ArgumentParser(description="TF-IDF search engine for .md files")
    parser.add_argument("folder", help="Path to folder containing .md files")
    args = parser.parse_args()

    engine = TFIDFSearch(args.folder)
    print("Ready. Type a query or :reload / :quit", file=sys.stderr)
    try:
        while True:
            query = input("> ").strip()
            if query == ":quit":
                break
            if query == ":reload":
                engine.reload()
                continue
            if not query:
                continue
            results = engine.search(query)
            if not results:
                print("  (no matches)")
            else:
                for i, (doc, score) in enumerate(results, 1):
                    print(f"  {i}. [{score:.4f}] {doc}")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
