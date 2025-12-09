#!/usr/bin/env python3
"""
LLM Hallucination Simulator
Sends queries designed to trigger hallucinations (out-of-domain, no context)
"""
import requests
import time
import argparse
import random
from typing import List, Dict

LLM_API_URL = "http://localhost:8001"

# Out-of-domain queries that won't have relevant context
OUT_OF_DOMAIN_QUERIES = [
    "What is the capital of France?",
    "Who won the World Cup in 2022?",
    "Explain quantum entanglement",
    "What's the recipe for chocolate cake?",
    "How do I train a neural network?",
    "What is the meaning of life?",
    "Explain the theory of relativity",
    "How does photosynthesis work?",
    "What's the weather like today?",
    "Tell me about ancient Rome",
    "How do I learn to play guitar?",
    "What are the best tourist destinations?",
    "Explain blockchain technology in simple terms",
    "How do I fix my car engine?",
    "What's the best programming language?",
]

# Ambiguous queries that might lead to hallucination
AMBIGUOUS_QUERIES = [
    "How do I fix the thing?",
    "It's not working, help!",
    "What should I do about the problem?",
    "Can you help me with that issue?",
    "Something is broken, how to fix?",
    "The system is down, what now?",
    "I need help with stuff",
    "Things aren't right, advice?",
]

# Some valid IT queries for contrast
VALID_QUERIES = [
    "How do I reset my password?",
    "I can't access my account",
    "VPN connection issues",
    "Software installation help",
    "Network connectivity problems",
]

def send_chat_request(query: str, session_id: str) -> Dict:
    """Send chat request to LLM API"""
    try:
        response = requests.post(
            f"{LLM_API_URL}/chat",
            json={
                "query": query,
                "session_id": session_id,
                "num_context_docs": 3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def check_embedding_drift() -> Dict:
    """Check embedding drift stats"""
    try:
        response = requests.get(f"{LLM_API_URL}/drift/embedding-stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error checking drift: {e}")
    return {}

def simulate_hallucinations(num_queries: int, hallucination_rate: float):
    """
    Simulate LLM hallucinations
    
    Args:
        num_queries: Number of queries to send
        hallucination_rate: Target hallucination rate (0.0-1.0)
    """
    print(f"Starting LLM Hallucination Simulation")
    print(f"Queries: {num_queries}, Target Hallucination Rate: {hallucination_rate}")
    print("=" * 60)
    
    total_queries = 0
    total_hallucination_score = 0.0
    high_hallucination_count = 0
    
    for i in range(num_queries):
        # Choose query type based on target hallucination rate
        rand_val = random.random()
        
        if rand_val < hallucination_rate * 0.6:
            # Out-of-domain query (high hallucination probability)
            query = random.choice(OUT_OF_DOMAIN_QUERIES)
            query_type = "out-of-domain"
        elif rand_val < hallucination_rate:
            # Ambiguous query (medium hallucination probability)
            query = random.choice(AMBIGUOUS_QUERIES)
            query_type = "ambiguous"
        else:
            # Valid query (low hallucination probability)
            query = random.choice(VALID_QUERIES)
            query_type = "valid"
        
        session_id = f"hallucination-sim-{i}"
        
        print(f"\n[Query {i+1}/{num_queries}] Type: {query_type}")
        print(f"Query: {query}")
        
        result = send_chat_request(query, session_id)
        
        if "error" not in result:
            hallucination_score = result.get("hallucination_score", 0.0)
            relevance_score = result.get("relevance_score", 0.0)
            groundedness_score = result.get("groundedness_score", 0.0)
            
            total_hallucination_score += hallucination_score
            total_queries += 1
            
            if hallucination_score > 0.7:
                high_hallucination_count += 1
                status = "🔴 HIGH HALLUCINATION"
            elif hallucination_score > 0.4:
                status = "🟡 MEDIUM"
            else:
                status = "🟢 LOW"
            
            print(f"{status}")
            print(f"  Hallucination: {hallucination_score:.3f}")
            print(f"  Relevance: {relevance_score:.3f}")
            print(f"  Groundedness: {groundedness_score:.3f}")
            print(f"  Response: {result.get('response', '')[:100]}...")
        else:
            print(f"❌ Error: {result.get('error')}")
        
        # Check embedding drift periodically
        if (i + 1) % 10 == 0:
            drift_stats = check_embedding_drift()
            print(f"\n--- Drift Stats (after {i+1} queries) ---")
            print(f"  Avg Quality: {drift_stats.get('avg_quality', 0):.3f}")
            print(f"  Embedding Drift: {drift_stats.get('drift_score', 0):.3f}")
            print(f"  Query Diversity: {drift_stats.get('diversity_score', 0):.3f}")
        
        time.sleep(1)  # Rate limiting
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"\nStatistics:")
    print(f"  Total Queries: {total_queries}")
    print(f"  Avg Hallucination Score: {total_hallucination_score/max(total_queries,1):.3f}")
    print(f"  High Hallucination Count: {high_hallucination_count} "
          f"({high_hallucination_count/max(total_queries,1)*100:.1f}%)")
    
    final_drift = check_embedding_drift()
    print(f"\nFinal Embedding Stats:")
    print(f"  Avg Quality Score: {final_drift.get('avg_quality', 0):.3f}")
    print(f"  Embedding Drift: {final_drift.get('drift_score', 0):.3f}")
    
    print(f"\nCheck Prometheus alerts: http://localhost:9090/alerts")
    print(f"Check Grafana dashboard: http://localhost:3000")
    print(f"View TruLens logs: docker compose exec llm-rag-api cat /logs/trulens/evaluations.log")

def main():
    parser = argparse.ArgumentParser(description="Simulate LLM hallucinations")
    parser.add_argument("--num-queries", type=int, default=30,
                       help="Number of queries to send (default: 30)")
    parser.add_argument("--hallucination-rate", type=float, default=0.7,
                       help="Target hallucination rate 0.0-1.0 (default: 0.7)")
    
    args = parser.parse_args()
    
    if not (0.0 <= args.hallucination_rate <= 1.0):
        print("Error: hallucination-rate must be between 0.0 and 1.0")
        return
    
    try:
        simulate_hallucinations(args.num_queries, args.hallucination_rate)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

if __name__ == "__main__":
    main()
