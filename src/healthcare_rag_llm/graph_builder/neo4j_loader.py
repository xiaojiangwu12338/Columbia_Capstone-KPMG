# src/healthcare_rag_llm/graph_builder/neo4j_loader.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# docker/.env
load_dotenv("docker/.env")

class Neo4jConnector:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        if not self.password:
            raise ValueError("ERROR. NEO4J_PASSWORD is not set. Please check your .env file.")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def init_schema(self):
        """Create uniqueness constraints and vector index"""
        with self.driver.session() as session:
            # Unique constraints
            session.run("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
                        "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            session.run("CREATE CONSTRAINT page_uid_unique IF NOT EXISTS "
                        "FOR (p:Page) REQUIRE p.uid IS UNIQUE")
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

            # Vector index (BGE-M3 output dim: 1024 )
            session.run("""
            CREATE VECTOR INDEX chunk_vec IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 1024,
              `vector.similarity_function`: 'cosine'
            }}
            """)
            print("Neo4j schema initialized.")
