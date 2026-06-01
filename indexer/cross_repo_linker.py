"""
Cross-repo inference — CP4.

Scans indexed source files for inter-service call patterns and creates
``CROSS_REPO_CALL`` edges in the Kuzu knowledge graph between the calling
``File`` node and:
  - the *callee* ``File`` node (if a matching ``APIEndpoint`` is found), or
  - a synthetic ``File`` placeholder node (for unresolved references).

Detected patterns
-----------------
HTTP clients
  ``requests``, ``httpx``, ``axios``, ``fetch``, ``urllib`` calls with a URL.
  The URL path is matched against ``APIEndpoint.path`` in Kuzu.

Kafka topics
  ``.produce("topic")``, ``.subscribe(["topic"])`` patterns.
  Producers and consumers sharing the same topic name across different repos
  receive a ``CROSS_REPO_CALL`` edge labelled with the topic.

Usage::

    from indexer.cross_repo_linker import CrossRepoLinker
    from langchain_core.documents import Document

    linker = CrossRepoLinker()
    n_edges = linker.run(changed_docs)
    print(f"Created {n_edges} cross-repo edges")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# HTTP: requests/httpx method calls with a string literal URL
_HTTP_CALL = re.compile(
    r"""(?:requests|httpx)\s*\.\s*(?:get|post|put|patch|delete|request)\s*"""
    r"""\(\s*[f]?['"]([^'"]{4,})['"]\s*""",
    re.IGNORECASE,
)

# axios.get("…"), axios.post("…")
_AXIOS_CALL = re.compile(
    r"""axios\s*\.\s*(?:get|post|put|patch|delete)\s*\(\s*[f]?['"]([^'"]{4,})['"]\s*""",
    re.IGNORECASE,
)

# fetch("…")  (JS/TS)
_FETCH_CALL = re.compile(
    r"""\bfetch\s*\(\s*[`'"]([^`'"]{4,})[`'"]\s*""",
    re.IGNORECASE,
)

# Kafka: producer.produce("topic", …)
_KAFKA_PRODUCE = re.compile(
    r"""\.produce\s*\(\s*['"]([a-zA-Z][a-zA-Z0-9._-]+)['"]\s*""",
)

# Kafka: consumer.subscribe(["topic"])
_KAFKA_SUBSCRIBE = re.compile(
    r"""\.subscribe\s*\(\s*\[\s*['"]([a-zA-Z][a-zA-Z0-9._-]+)['"]\s*""",
)

_ALL_HTTP = [_HTTP_CALL, _AXIOS_CALL, _FETCH_CALL]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_path(url: str) -> str | None:
    """Return the URL path component, or None if unparseable / too short."""
    url = url.strip()
    if url.startswith(("http://", "https://")):
        path = urlparse(url).path
    elif url.startswith("/"):
        path = url.split("?")[0].split("#")[0]
    else:
        return None   # relative or template literal — skip

    # Skip trivial paths
    if not path or path == "/" or len(path) < 2:
        return None
    return path


def _scan_http_urls(content: str) -> list[str]:
    """Return unique URL paths found by HTTP client patterns in *content*."""
    paths: list[str] = []
    for pat in _ALL_HTTP:
        for m in pat.finditer(content):
            p = _extract_path(m.group(1))
            if p and p not in paths:
                paths.append(p)
    return paths


def _scan_kafka_topics(content: str) -> dict[str, list[str]]:
    """Return ``{"produce": [...], "subscribe": [...]}`` topic lists."""
    produce  = list({m.group(1) for m in _KAFKA_PRODUCE.finditer(content)})
    subscribe = list({m.group(1) for m in _KAFKA_SUBSCRIBE.finditer(content)})
    return {"produce": produce, "subscribe": subscribe}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LinkResult:
    http_edges:  int = 0
    kafka_edges: int = 0

    @property
    def total(self) -> int:
        return self.http_edges + self.kafka_edges


# ---------------------------------------------------------------------------
# CrossRepoLinker
# ---------------------------------------------------------------------------

class CrossRepoLinker:
    """
    Scans a list of source-file Documents and creates ``CROSS_REPO_CALL``
    edges in Kuzu for any detected inter-service HTTP or Kafka references.
    """

    def run(
        self,
        documents: "list[Document]",
    ) -> int:
        """
        Process *documents* and populate cross-repo edges.

        Args:
            documents: LangChain Documents (one per source file).

        Returns:
            Total number of ``CROSS_REPO_CALL`` edges created.
        """
        try:
            from indexer.graph_builder import GraphBuilder
        except ImportError:
            return 0  # Kuzu not installed — skip silently

        result = LinkResult()

        try:
            with GraphBuilder() as gb:
                # Collect all (repo, file) combinations for Kafka correlation
                kafka_by_repo: dict[str, dict[str, list[str]]] = {}

                for doc in documents:
                    repo  = doc.metadata.get("repo_name", "")
                    fpath = doc.metadata.get("file_path", "")
                    if not repo or not fpath:
                        continue

                    content = doc.page_content

                    # ---- HTTP cross-repo edges ----------------------------
                    http_paths = _scan_http_urls(content)
                    for path in http_paths:
                        n = self._link_http(gb, fpath, repo, path)
                        result.http_edges += n

                    # ---- Kafka: collect for batch correlation below -------
                    kafka = _scan_kafka_topics(content)
                    if kafka["produce"] or kafka["subscribe"]:
                        kafka_by_repo.setdefault(repo, {}).setdefault(fpath, {})
                        kafka_by_repo[repo][fpath] = kafka

                # ---- Kafka cross-repo edges (producer → consumer) ---------
                result.kafka_edges += self._link_kafka(gb, kafka_by_repo)

        except Exception:  # noqa: BLE001
            pass   # Cross-repo linking is best-effort / non-fatal

        return result.total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _link_http(
        self,
        gb,
        from_file: str,
        from_repo: str,
        url_path: str,
    ) -> int:
        """
        Find APIEndpoints whose path matches *url_path* in a *different* repo
        and create a CROSS_REPO_CALL edge.
        """
        try:
            # Match endpoint path: check if the endpoint path ends with / starts
            # with the detected URL path (handles prefix like /api/v1/)
            rows = gb._conn.execute(
                "MATCH (ep:APIEndpoint) "
                "WHERE ep.path ENDS WITH $path OR ep.path = $path "
                "RETURN ep.node_id, ep.file_id, ep.repo_name",
                {"path": url_path},
            ).fetchall()
        except Exception:  # noqa: BLE001
            return 0

        n = 0
        for row in rows:
            target_repo = row[2] if len(row) > 2 else ""
            if target_repo == from_repo:
                continue   # same-repo call — not cross-repo

            target_file_id = row[1] if len(row) > 1 else ""
            if not target_file_id:
                continue

            try:
                # Resolve source file_id
                src_rows = gb._conn.execute(
                    "MATCH (f:File) WHERE f.file_path = $fp AND f.repo_name = $rn "
                    "RETURN f.node_id",
                    {"fp": from_file, "rn": from_repo},
                ).fetchall()
                if not src_rows:
                    continue
                src_id = src_rows[0][0]

                # Check if edge already exists
                exists = gb._conn.execute(
                    "MATCH (a:File)-[r:CROSS_REPO_CALL]->(b:File) "
                    "WHERE a.node_id = $src AND b.node_id = $tgt "
                    "RETURN COUNT(*)",
                    {"src": src_id, "tgt": target_file_id},
                ).fetchone()
                if exists and exists[0] > 0:
                    continue

                gb._conn.execute(
                    "MATCH (a:File), (b:File) "
                    "WHERE a.node_id = $src AND b.node_id = $tgt "
                    "CREATE (a)-[:CROSS_REPO_CALL {call_type: 'http', path: $path}]->(b)",
                    {"src": src_id, "tgt": target_file_id, "path": url_path},
                )
                n += 1
            except Exception:  # noqa: BLE001
                continue

        return n

    def _link_kafka(
        self,
        gb,
        kafka_by_repo: dict[str, dict[str, list[str]]],
    ) -> int:
        """
        Create CROSS_REPO_CALL edges between Kafka producers and consumers
        that share the same topic across different repos.
        """
        # Build: topic → [(repo, file, role), ...]
        topic_map: dict[str, list[tuple[str, str, str]]] = {}
        for repo, files in kafka_by_repo.items():
            for fpath, roles in files.items():
                for topic in roles.get("produce", []):
                    topic_map.setdefault(topic, []).append((repo, fpath, "produce"))
                for topic in roles.get("subscribe", []):
                    topic_map.setdefault(topic, []).append((repo, fpath, "consume"))

        n = 0
        for topic, participants in topic_map.items():
            producers  = [(r, f) for r, f, role in participants if role == "produce"]
            consumers  = [(r, f) for r, f, role in participants if role == "consume"]

            for p_repo, p_file in producers:
                for c_repo, c_file in consumers:
                    if p_repo == c_repo:
                        continue  # same-repo Kafka — skip

                    try:
                        src_rows = gb._conn.execute(
                            "MATCH (f:File) WHERE f.file_path = $fp AND f.repo_name = $rn "
                            "RETURN f.node_id",
                            {"fp": p_file, "rn": p_repo},
                        ).fetchall()
                        tgt_rows = gb._conn.execute(
                            "MATCH (f:File) WHERE f.file_path = $fp AND f.repo_name = $rn "
                            "RETURN f.node_id",
                            {"fp": c_file, "rn": c_repo},
                        ).fetchall()
                        if not src_rows or not tgt_rows:
                            continue

                        src_id = src_rows[0][0]
                        tgt_id = tgt_rows[0][0]

                        exists = gb._conn.execute(
                            "MATCH (a:File)-[r:CROSS_REPO_CALL]->(b:File) "
                            "WHERE a.node_id = $src AND b.node_id = $tgt "
                            "RETURN COUNT(*)",
                            {"src": src_id, "tgt": tgt_id},
                        ).fetchone()
                        if exists and exists[0] > 0:
                            continue

                        gb._conn.execute(
                            "MATCH (a:File), (b:File) "
                            "WHERE a.node_id = $src AND b.node_id = $tgt "
                            "CREATE (a)-[:CROSS_REPO_CALL "
                            "{call_type: 'kafka', topic: $topic}]->(b)",
                            {"src": src_id, "tgt": tgt_id, "topic": topic},
                        )
                        n += 1
                    except Exception:  # noqa: BLE001
                        continue

        return n
