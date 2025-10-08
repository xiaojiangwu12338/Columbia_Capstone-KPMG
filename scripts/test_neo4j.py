'''
### Steps
## 1. Install Docker
Download and install Docker Desktop from the official site: 
https://www.docker.com/products/docker-desktop/

Make sure Docker is running before continuing.


## 2. Configure the Environment
Go to the project folder’s docker directory.
Open the sample environment file (docker/.env.example).
Create a new environment file (docker/.env) based on it (instructions noted in the sample environment file).


## 3. Start the Neo4j Database
```
cd docker
docker compose up -d
```
This will:
- Start a Neo4j container
- Expose the Neo4j Browser at http://localhost:7474
- Log in using the credentials from .env

To verify the container is running:
```
docker ps
```


## 4. Test the Connection
```
cd ..
python scripts/test_neo4j.py
```


## 5. Ingest Documents and Build the Graph
To load processed chunks into Neo4j:
```
python scripts/ingest_graph.py --chunk_dir data/chunks/asterisk_separate_chunking_result
```


## 6. Query the Graph
- Option 1: Browser UI (recommended for visualization)
Open http://localhost:7474 and run queries like:
```
MATCH (d:Document)-[:CONTAINS]->(p:Page)-[:HAS_CHUNK]->(c:Chunk)
RETURN d.doc_id, p.page_no, c.chunk_id
LIMIT 10;
```
- Option 2: Python test case queries
```
python src/healthcare_rag_llm/graph_builder/queries.py
```


## 7. Shut Down or Reset the Database
To stop the Neo4j container while keeping data:
```
cd docker
docker compose down
```

To fully reset (remove all data and restart cleanly)
- Option 1: Reset only graph data (keep schema and container)
Delete all nodes and relationships in Neo4j without removing the database container or indexes:
```
python scripts/reset_graph.py
```
- Option 2: Reset Docker container and all stored data
Completely remove all stored data, indexes, and settings (start from a clean database):
```
cd docker
docker compose down -v
```
'''

# test if neo4j connected
from neo4j import GraphDatabase
import os

# load from .env (can use python-dotenv to automatically load .env in the future)
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "healthragkpmg")

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    result = session.run("RETURN 'Neo4j connected!' AS msg")
    print(result.single()["msg"])

driver.close()