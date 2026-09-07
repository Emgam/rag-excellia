import requests
import time
import textwrap

API_URL = "http://localhost:8000/query"

TEST_CASES = [
    ("Compare the screen sizes available in First Class (CloudLux) and Economy Class (CloudSaver) on the Airbus A380.", ["22", "10"], "complex"),
    ("What are the step-by-step requirements to open a locker at ZX Bank?", ["kyc", "application", "rent", "keys"], "complex"),
    ("What are the different channels I can use to request the closure of a ZX Bank Credit Card?", ["app", "netbanking", "customer care", "branch"], "complex"),
    ("Who is eligible for complimentary accommodation during layovers at CloudWay 24, and what standard of hotels are provided?", ["pilots", "crew", "4-star"], "complex"),
    
    ("What is the seating capacity of the Boeing 747 in CloudWay 24?", [], "hallucination"),
    ("What is the weather like in Paris today?", [], "hallucination"),
    ("What is the recipe for a chocolate cake?", [], "hallucination"),
    ("Who won the FIFA World Cup in 2022?", [], "hallucination")
]

REFUSAL_KEYWORDS = ["could not find", "not found", "not specified", "apologize", "do not contain", "unable to find", "does not have", "not listed"]

def run_tests():
    print("=" * 70)
    print("🧪 EXCELLIA RAG - TESTS DE COMPLEXITÉ ET D'HALLUCINATION (FINAL)")
    print("=" * 70)
    
    stats = {"complex_total": 0, "complex_success": 0, "complex_time": 0, "hallucination_total": 0, "hallucination_success": 0}
    
    for i, (question, expected_keywords, test_type) in enumerate(TEST_CASES, 1):
        print(f"\n--- Question {i}/{len(TEST_CASES)} [{test_type.upper()}] ---")
        print(f"❓ {question}")
        
        payload = {"question": question, "top_k": 3, "stream": False}
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            continue
            
        elapsed_time = time.time() - start_time
        answer = data.get("answer", "")
        is_reliable = data.get("reliable", False)
        
        wrapped_answer = textwrap.fill(answer, width=80, initial_indent="💬 ", subsequent_indent="   ")
        print(wrapped_answer)
        print(f"⏱️ Temps: {elapsed_time:.2f}s | Fiable: {is_reliable}")
        
        answer_lower = answer.lower()
        
        if test_type == "complex":
            stats["complex_total"] += 1
            stats["complex_time"] += elapsed_time
            if any(kw.lower() in answer_lower for kw in expected_keywords) and is_reliable:
                print("✅ RÉSULTAT: SUCCÈS (Réponse pertinente)")
                stats["complex_success"] += 1
            else:
                print("❌ RÉSULTAT: ÉCHEC (Mots-clés manquants)")
                
        elif test_type == "hallucination":
            stats["hallucination_total"] += 1
            has_refused = any(kw in answer_lower for kw in REFUSAL_KEYWORDS) or not is_reliable
            if has_refused:
                print("✅ RÉSULTAT: SUCCÈS (Hallucination évitée)")
                stats["hallucination_success"] += 1
            else:
                print("❌ RÉSULTAT: ÉCHEC (Hallucination détectée !)")
                
    print("\n" + "=" * 70)
    print("📊 RAPPORT FINAL DES TESTS")
    print("=" * 70)
    if stats["complex_total"] > 0:
        acc = (stats["complex_success"] / stats["complex_total"]) * 100
        avg_time = stats["complex_time"] / stats["complex_total"]
        print(f"🎯 Précision Questions Complexes : {acc:.1f}% ({stats['complex_success']}/{stats['complex_total']})")
        print(f"⏱️ Temps Moyen (Complexes)      : {avg_time:.2f} secondes")
    if stats["hallucination_total"] > 0:
        safety = (stats["hallucination_success"] / stats["hallucination_total"]) * 100
        print(f"🛡️ Taux de Sécurité (Anti-Hallucination) : {safety:.1f}% ({stats['hallucination_success']}/{stats['hallucination_total']})")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()