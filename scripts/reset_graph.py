# scripts/reset_graph.py
"""
Usage:
    python scripts/reset_graph.py
Clear all nodes and relationships in Neo4j (**irreversible operation**)

Steps:
1. Run the following command in the project root directory:
```
python scripts/reset_graph.py
```

You will be prompted:
"This will DELETE ALL nodes and relationships in Neo4j. Continue? (y/N):"

2. Enter y → All nodes and edges in Neo4j will be cleared.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector


def reset_graph():
    connector = Neo4jConnector()
    with connector.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    connector.close()
    print("All nodes and relationships deleted from Neo4j.")


if __name__ == "__main__":
    confirm = input("This will DELETE ALL nodes and relationships in Neo4j. Continue? (y/N): ")
    if confirm.lower() == "y":
        reset_graph()
    else:
        print("Cancelled.")