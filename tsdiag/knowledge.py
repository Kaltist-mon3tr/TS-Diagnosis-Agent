from __future__ import annotations

import json
import os
from pathlib import Path


class KnowledgeRetriever:
    """Read-only retrieval from a local JSON graph or an optional local Neo4j DB."""

    def __init__(self, json_path: str | Path = "knowledge/graph.json", use_neo4j: bool = False, database: str = "neo4j"):
        self.json_path = Path(json_path)
        self.use_neo4j = use_neo4j
        self.database = database

    def retrieve(self, query_terms: list[str], limit: int = 8) -> list[dict]:
        terms = [term.casefold().strip() for term in query_terms if term and term.strip()]
        if self.use_neo4j:
            return self._retrieve_neo4j(terms, limit)
        with self.json_path.open("r", encoding="utf-8") as handle:
            graph = json.load(handle)
        ranked = []
        for node in graph.get("nodes", []):
            searchable = " ".join([
                str(node.get("name", "")), str(node.get("type", "")),
                str(node.get("description", "")), " ".join(node.get("keywords", [])),
            ]).casefold()
            matches = [term for term in terms if term in searchable]
            if matches or node.get("type") == "general":
                ranked.append((len(matches), node))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": node.get("id"), "name": node.get("name"), "type": node.get("type"),
                "description": node.get("description"), "actions": node.get("actions", []),
                "source": node.get("source", "unspecified_local_knowledge"),
                "source_verified": bool(node.get("source_verified", False)),
            }
            for _, node in ranked[:limit]
        ]

    def _retrieve_neo4j(self, terms: list[str], limit: int) -> list[dict]:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j retrieval requested; install the neo4j Python driver from environment.yml") from exc
        uri = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise RuntimeError("Set NEO4J_PASSWORD in the current environment before using --neo4j")
        query = """
        MATCH (n:KGNode)
        WHERE any(term IN $terms WHERE
          toLower(coalesce(n.name, '')) CONTAINS term OR
          toLower(coalesce(n.description, '')) CONTAINS term OR
          any(k IN coalesce(n.keywords, []) WHERE toLower(k) CONTAINS term))
        OPTIONAL MATCH (n)-[r:RELATED]->(m:KGNode)
        RETURN properties(n) AS node,
               collect({relation: r.relation, target: properties(m)})[0..8] AS related
        LIMIT $limit
        """
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            records, _, _ = driver.execute_query(query, terms=terms, limit=limit, database_=self.database)
        return [
            {
                "id": record["node"].get("id"), "name": record["node"].get("name"),
                "type": record["node"].get("type"), "description": record["node"].get("description"),
                "keywords": record["node"].get("keywords", []), "related": record["related"],
                "source": "local_neo4j",
                "source_verified": bool(record["node"].get("source_verified", False)),
            }
            for record in records
        ]

