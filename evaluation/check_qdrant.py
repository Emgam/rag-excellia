import requests
import json

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "excellia_docs"

def check_chunk_sizes():
    print(f"🔍 Fetching sample chunks from Qdrant collection: {COLLECTION_NAME}...")
    
    # Scroll API gets points without needing a vector search
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/scroll"
    payload = {
        "limit": 5,
        "with_payload": True,   # Fixed: Capitalized True
        "with_vector": False    # Fixed: Capitalized False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        points = data.get("result", {}).get("points", [])
        
        if not points:
            print("❌ No points found in collection.")
            return
            
        print("\n📊 Sample Chunk Sizes (in characters):")
        print("-" * 50)
        
        for i, point in enumerate(points):
            payload_data = point.get("payload", {})
            text = payload_data.get("text", "")
            char_count = len(text)
            # Rough estimate: 4 characters = 1 token
            token_est = char_count / 4 
            
            print(f"Chunk {i+1}: {char_count} chars (~{int(token_est)} tokens)")
            print(f"   Text preview: {text[:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    check_chunk_sizes()