"""
Quick sanity check for Neo4j graph content.

Usage:
    python src/healthcare_rag_llm/graph_builder/graph_summary.py
Output:
    Prints node/relationship counts and chunk type breakdown.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector

def check_graph_summary():
    connector = Neo4jConnector()

    with connector.driver.session() as session:
        # --- Node counts ---
        result = session.run("""
        CALL {
            MATCH (a:Authority) RETURN 'Authority' AS Label, count(a) AS Count
            UNION
            MATCH (d:Document) RETURN 'Document' AS Label, count(d) AS Count
            UNION
            MATCH (p:Page) RETURN 'Page' AS Label, count(p) AS Count
            UNION
            MATCH (c:Chunk) RETURN 'Chunk (Total)' AS Label, count(c) AS Count
            UNION
            MATCH (c:Chunk {type:'text'}) RETURN 'Chunk: text' AS Label, count(c) AS Count
            UNION
            MATCH (c:Chunk {type:'table'}) RETURN 'Chunk: table' AS Label, count(c) AS Count
            UNION
            MATCH (c:Chunk {type:'ocr'}) RETURN 'Chunk: ocr' AS Label, count(c) AS Count
        }
        RETURN Label, Count
        """)
        print("=== Node Summary ===")
        for row in result:
            print(f"{row['Label']:<20} {row['Count']}")

        # --- Relationship counts ---
        rel_result = session.run("""
        MATCH ()-[r]->() 
        RETURN type(r) AS Relationship, count(*) AS Count
        ORDER BY Count DESC
        """)
        print("\n=== Relationship Summary ===")
        for row in rel_result:
            print(f"{row['Relationship']:<20} {row['Count']}")

    connector.close()

if __name__ == "__main__":
    check_graph_summary()
