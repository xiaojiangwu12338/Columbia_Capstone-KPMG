'''
####
Steps
1. Install Docker:
https://www.docker.com/products/docker-desktop/

2. Start the database
```
cd docker
docker compose up -d
```
Then visit: http://localhost:7474

3. Test connection
```
cd ..
python scripts/test_neo4j.py
```

4. Shut down the database
```
cd docker
docker compose down
```
or reset database
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