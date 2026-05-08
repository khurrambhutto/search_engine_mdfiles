#!/usr/bin/env python3
import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TFIDFSearch:
    def __init__(self, folder: str, mode: str = "tfidf", k1: float = 1.5, b: float = 0.75):
        self.folder = Path(folder)
        self.mode = mode
        self.k1 = k1
        self.b = b
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.docs = {}            # doc_name -> [lemmatized tokens]
        self.vocab = set()
        self.term_counts = {}     # doc_name -> {word: raw count}
        self.doc_lengths = {}     # doc_name -> total tokens
        self.idf = {}             # word -> idf
        self.tfidf = {}           # doc_name -> {word: tfidf}  (for tfidf mode)
        self._build_index()

    def _strip_markdown(self, text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`[^`]+`", " ", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"^#{1,6}\s*", " ", text, flags=re.M)
        text = re.sub(r"\*{1,3}|_{1,3}|~~|==|\[\^[^\]]+\]", " ", text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        text = self._strip_markdown(text)
        tokens = re.findall(r"[a-z]{2,}", text.lower())
        tokens = [t for t in tokens if t not in self.stopwords]
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def _load_docs(self):
        for md_file in self.folder.rglob("*.md"):
            name = str(md_file.relative_to(self.folder))
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            tokens = self._tokenize(text)
            self.docs[name] = tokens
            self.doc_lengths[name] = len(tokens)
            counts = defaultdict(int)
            for t in tokens:
                counts[t] += 1
                self.vocab.add(t)
            self.term_counts[name] = dict(counts)

    def _compute_idf(self):
        n = len(self.docs)
        for word in self.vocab:
            df = sum(1 for counts in self.term_counts.values() if word in counts)
            self.idf[word] = math.log(n / (1 + df)) + 1

    def _compute_tfidf(self):
        for doc in self.docs:
            self.tfidf[doc] = {}
            length = self.doc_lengths[doc]
            if length == 0:
                continue
            for word, count in self.term_counts[doc].items():
                self.tfidf[doc][word] = (count / length) * self.idf[word]

    def _build_index(self):
        self._load_docs()
        self._compute_idf()
        self._compute_tfidf()
        print(f"Indexed {len(self.docs)} documents, {len(self.vocab)} unique terms "
              f"({self.mode} mode).", file=sys.stderr)

    def _score_tfidf(self, query: str) -> dict[str, float]:
        qtokens = self._tokenize(query)
        if not qtokens:
            return {}
        counts = defaultdict(int)
        for t in qtokens:
            counts[t] += 1
        qvec = {w: (c / len(qtokens)) * self.idf.get(w, 0) for w, c in counts.items()}

        scores = {}
        for doc, dvec in self.tfidf.items():
            if not any(w in dvec for w in qvec):
                continue
            dot = sum(qvec.get(w, 0) * dvec.get(w, 0) for w in qvec)
            norm_q = math.sqrt(sum(v ** 2 for v in qvec.values()))
            norm_d = math.sqrt(sum(v ** 2 for v in dvec.values()))
            if norm_q > 0 and norm_d > 0:
                scores[doc] = dot / (norm_q * norm_d)
        return scores

    def _score_bm25(self, query: str) -> dict[str, float]:
        qtokens = self._tokenize(query)
        if not qtokens:
            return {}
        avgdl = sum(self.doc_lengths.values()) / max(len(self.docs), 1)

        scores = {}
        for doc in self.docs:
            length = self.doc_lengths[doc]
            if length == 0:
                continue
            counts = self.term_counts[doc]
            score = 0.0
            for t in qtokens:
                idf = self.idf.get(t, 0)
                f = counts.get(t, 0)
                if f == 0:
                    continue
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * length / avgdl)
                score += idf * numerator / denominator
            if score > 0:
                scores[doc] = score
        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self.mode == "bm25":
            scores = self._score_bm25(query)
        else:
            scores = self._score_tfidf(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def reload(self):
        self.docs.clear()
        self.vocab.clear()
        self.term_counts.clear()
        self.doc_lengths.clear()
        self.idf.clear()
        self.tfidf.clear()
        self._build_index()


def main():
    parser = argparse.ArgumentParser(description="TF-IDF / BM25 search engine for .md files")
    parser.add_argument("folder", help="Path to folder containing .md files")
    parser.add_argument("--bm25", action="store_true", help="Use BM25 scoring instead of TF-IDF")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter (default: 1.5)")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter (default: 0.75)")
    args = parser.parse_args()

    mode = "bm25" if args.bm25 else "tfidf"
    engine = TFIDFSearch(args.folder, mode=mode, k1=args.k1, b=args.b)
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
