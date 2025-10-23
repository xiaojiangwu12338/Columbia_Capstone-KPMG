# src/healthcare_rag_llm/graph_builder/neo4j_loader.py
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load Docker .env
load_dotenv("docker/.env")

class Neo4jConnector:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")

        # NEO4J_AUTH
        auth = os.getenv("NEO4J_AUTH")
        if auth and "/" in auth:
            user_env, password_env = auth.split("/", 1)
        else:
            user_env = os.getenv("NEO4J_USERNAME", "neo4j")
            password_env = os.getenv("NEO4J_PASSWORD")

        self.user = user or user_env
        self.password = password or password_env

        if not self.password:
            raise ValueError("ERROR: Neo4j password not set. Check your .env file.")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def init_schema(self):
        """Initialize Neo4j constraints and vector index"""
        with self.driver.session() as session:
            # Unique constraints
            session.run("CREATE CONSTRAINT authority_name_unique IF NOT EXISTS "
                        "FOR (a:Authority) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
                        "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            session.run("CREATE CONSTRAINT page_uid_unique IF NOT EXISTS "
                        "FOR (p:Page) REQUIRE p.uid IS UNIQUE")
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

            # Vector index for dense embedding (BGE-M3 → 1024 dims)
            session.run("""
            CREATE VECTOR INDEX chunk_vec IF NOT EXISTS
            FOR (c:Chunk) ON (c.denseEmbedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 1024,
              `vector.similarity_function`: 'cosine'
            }}
            """)
            print("✅ Neo4j schema initialized (Authority, Document, Page, Chunk, vector index).")
