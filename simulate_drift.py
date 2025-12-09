#!/usr/bin/env python3
"""
ML Data Drift Simulator
Sends requests with shifted feature distributions to trigger drift detection
"""
import requests
import time
import argparse
import random
import numpy as np
from typing import Dict

ML_API_URL = "http://localhost:8000"

def generate_normal_request() -> Dict:
    """Generate a request with normal distribution"""
    return {
        "ticket_length": int(np.random.normal(120, 40)),
        "urgency_keywords": int(np.clip(np.random.poisson(2), 0, 10)),
        "business_impact": float(np.clip(np.random.normal(5, 2), 0, 10)),
        "customer_tier": int(np.clip(np.random.normal(3, 1), 1, 5))
    }

def generate_drifted_request(drift_amount: float) -> Dict:
    """Generate a request with drifted distribution"""
    # Shift distributions significantly
    return {
        "ticket_length": int(np.random.normal(200 * drift_amount + 80 * (1-drift_amount), 50)),
        "urgency_keywords": int(np.clip(np.random.poisson(5 * drift_amount + 2 * (1-drift_amount)), 0, 15)),
        "business_impact": float(np.clip(np.random.normal(8 * drift_amount + 5 * (1-drift_amount), 1.5), 0, 10)),
        "customer_tier": int(np.clip(np.random.normal(4.5 * drift_amount + 3 * (1-drift_amount), 0.8), 1, 5))
    }

def send_prediction(request_data: Dict) -> bool:
    """Send prediction request"""
    try:
        response = requests.post(
            f"{ML_API_URL}/predict",
            json=request_data,
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending request: {e}")
        return False

def check_drift_status() -> Dict:
    """Check current drift status"""
    try:
        response = requests.get(f"{ML_API_URL}/drift/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error checking drift: {e}")
    return {}

def simulate_drift(duration: int, drift_amount: float, interval: float = 0.5):
    """
    Simulate data drift
    
    Args:
        duration: Simulation duration in seconds
        drift_amount: Drift magnitude (0.0-1.0)
        interval: Request interval in seconds
    """
    print(f"Starting ML Drift Simulation")
    print(f"Duration: {duration}s, Drift Amount: {drift_amount}, Interval: {interval}s")
    print("=" * 60)
    
    start_time = time.time()
    requests_sent = 0
    requests_success = 0
    
    # Send some normal requests first to establish baseline
    print("\n[Phase 1] Sending normal baseline traffic...")
    baseline_duration = min(30, duration // 4)
    baseline_end = start_time + baseline_duration
    
    while time.time() < baseline_end:
        request = generate_normal_request()
        if send_prediction(request):
            requests_success += 1
        requests_sent += 1
        
        if requests_sent % 10 == 0:
            drift_status = check_drift_status()
            print(f"Baseline: {requests_sent} requests, "
                  f"Drift Score: {drift_status.get('drift_score', 0):.4f}")
        
        time.sleep(interval)
    
    # Now send drifted requests
    print(f"\n[Phase 2] Injecting drift (amount={drift_amount})...")
    end_time = start_time + duration
    
    while time.time() < end_time:
        request = generate_drifted_request(drift_amount)
        if send_prediction(request):
            requests_success += 1
        requests_sent += 1
        
        if requests_sent % 10 == 0:
            drift_status = check_drift_status()
            drift_score = drift_status.get('drift_score', 0)
            drift_detected = drift_status.get('drift_detected', False)
            
            status_icon = "🔴" if drift_detected else "🟢"
            print(f"{status_icon} Sent: {requests_sent}, "
                  f"Success: {requests_success}, "
                  f"Drift Score: {drift_score:.4f}, "
                  f"Detected: {drift_detected}")
        
        time.sleep(interval)
    
    # Final status
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"Total Requests: {requests_sent}")
    print(f"Success Rate: {requests_success/requests_sent*100:.1f}%")
    
    final_status = check_drift_status()
    print(f"\nFinal Drift Status:")
    print(f"  Drift Score: {final_status.get('drift_score', 0):.4f}")
    print(f"  Drift Detected: {final_status.get('drift_detected', False)}")
    print(f"  Samples Analyzed: {final_status.get('samples_analyzed', 0)}")
    print(f"\nCheck Prometheus alerts: http://localhost:9090/alerts")
    print(f"Check Grafana dashboard: http://localhost:3000")

def main():
    parser = argparse.ArgumentParser(description="Simulate ML data drift")
    parser.add_argument("--duration", type=int, default=120,
                       help="Simulation duration in seconds (default: 120)")
    parser.add_argument("--drift-amount", type=float, default=0.7,
                       help="Drift magnitude 0.0-1.0 (default: 0.7)")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="Request interval in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    if not (0.0 <= args.drift_amount <= 1.0):
        print("Error: drift-amount must be between 0.0 and 1.0")
        return
    
    try:
        simulate_drift(args.duration, args.drift_amount, args.interval)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        print("Checking final drift status...")
        final_status = check_drift_status()
        print(f"Drift Score: {final_status.get('drift_score', 0):.4f}")

if __name__ == "__main__":
    main()
