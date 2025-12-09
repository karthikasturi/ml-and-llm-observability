#!/usr/bin/env python3
"""
Latency Spike Simulator
Sends burst traffic to create latency spikes
"""
import requests
import time
import argparse
import concurrent.futures
from typing import List

ML_API_URL = "http://localhost:8000"
LLM_API_URL = "http://localhost:8001"

def send_triage_request() -> float:
    """Send triage request and return latency"""
    request_data = {
        "ticket_text": "System performance degradation observed in production environment",
        "ticket_length": 150,
        "urgency_keywords": 3,
        "business_impact": 7,
        "customer_tier": 4
    }
    
    start = time.time()
    try:
        response = requests.post(
            f"{ML_API_URL}/triage",
            json=request_data,
            timeout=60
        )
        latency = (time.time() - start) * 1000  # ms
        return latency if response.status_code == 200 else -1
    except Exception as e:
        return -1

def send_concurrent_requests(num_requests: int, max_workers: int = 10) -> List[float]:
    """Send multiple concurrent requests"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_triage_request) for _ in range(num_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return [r for r in results if r > 0]

def simulate_latency_spike(duration: int, burst_size: int = 20, burst_interval: int = 5):
    """
    Simulate latency spikes through burst traffic
    
    Args:
        duration: Simulation duration in seconds
        burst_size: Number of concurrent requests per burst
        burst_interval: Seconds between bursts
    """
    print(f"Starting Latency Spike Simulation")
    print(f"Duration: {duration}s, Burst Size: {burst_size}, Interval: {burst_interval}s")
    print("=" * 60)
    
    start_time = time.time()
    burst_count = 0
    all_latencies = []
    
    while time.time() - start_time < duration:
        burst_count += 1
        print(f"\n🔥 Burst {burst_count}: Sending {burst_size} concurrent requests...")
        
        burst_start = time.time()
        latencies = send_concurrent_requests(burst_size)
        burst_duration = time.time() - burst_start
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            all_latencies.extend(latencies)
            
            print(f"  ✓ Completed in {burst_duration:.2f}s")
            print(f"  Success: {len(latencies)}/{burst_size}")
            print(f"  Avg Latency: {avg_latency:.0f}ms")
            print(f"  Min Latency: {min_latency:.0f}ms")
            print(f"  Max Latency: {max_latency:.0f}ms")
            
            if max_latency > 5000:
                print(f"  🔴 Alert threshold exceeded (>5000ms)!")
        else:
            print(f"  ❌ All requests failed")
        
        # Wait before next burst
        elapsed = time.time() - start_time
        if elapsed < duration:
            sleep_time = min(burst_interval, duration - elapsed)
            print(f"  Waiting {sleep_time:.0f}s before next burst...")
            time.sleep(sleep_time)
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    
    if all_latencies:
        all_latencies.sort()
        n = len(all_latencies)
        
        print(f"\nOverall Statistics ({n} successful requests):")
        print(f"  Total Bursts: {burst_count}")
        print(f"  Avg Latency: {sum(all_latencies)/n:.0f}ms")
        print(f"  Median (p50): {all_latencies[n//2]:.0f}ms")
        print(f"  p95: {all_latencies[int(n*0.95)]:.0f}ms")
        print(f"  p99: {all_latencies[int(n*0.99)]:.0f}ms")
        print(f"  Max: {max(all_latencies):.0f}ms")
    
    print(f"\nCheck Prometheus alerts: http://localhost:9090/alerts")
    print(f"Check Grafana latency dashboard: http://localhost:3000")

def main():
    parser = argparse.ArgumentParser(description="Simulate latency spikes")
    parser.add_argument("--duration", type=int, default=60,
                       help="Simulation duration in seconds (default: 60)")
    parser.add_argument("--burst-size", type=int, default=20,
                       help="Concurrent requests per burst (default: 20)")
    parser.add_argument("--burst-interval", type=int, default=5,
                       help="Seconds between bursts (default: 5)")
    
    args = parser.parse_args()
    
    try:
        simulate_latency_spike(args.duration, args.burst_size, args.burst_interval)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

if __name__ == "__main__":
    main()
