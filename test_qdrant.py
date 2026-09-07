from qdrant_client import QdrantClient

client = QdrantClient(host="127.0.0.1", port=6333)
try:
    print("Getting collections...")
    collections = client.get_collections()
    print(f"Collections: {collections}")
except Exception as e:
    print(f"Error: {e}")