"""
ML Inference API with Prometheus Metrics
Stage 2: Complete implementation
"""
import os
import time
import logging
from typing import Dict, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, 
    CONTENT_TYPE_LATEST, REGISTRY
)
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ML Inference API",
    description="Customer ticket severity prediction with Prometheus metrics",
    version="2.0.0"
)

# Prometheus Metrics
ml_prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Time spent processing prediction',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

ml_predictions_total = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['severity', 'status']
)

ml_prediction_errors = Counter(
    'ml_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

ml_model_info = Gauge(
    'ml_model_info',
    'ML model information',
    ['model_name', 'version', 'accuracy']
)

# Request and response models
class PredictionRequest(BaseModel):
    ticket_length: int = Field(..., ge=0, le=1000, description="Number of words in ticket")
    urgency_keywords: int = Field(..., ge=0, le=20, description="Count of urgency keywords")
    business_impact: float = Field(..., ge=0, le=10, description="Business impact score")
    customer_tier: int = Field(..., ge=1, le=5, description="Customer tier (1-5)")

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_length": 150,
                "urgency_keywords": 3,
                "business_impact": 7.0,
                "customer_tier": 4
            }
        }

class PredictionResponse(BaseModel):
    severity: str
    severity_code: int
    confidence: float
    processing_time_ms: float
    model_version: str = "1.0.0"

# Global model container
class ModelContainer:
    def __init__(self):
        self.model = None
        self.model_path = Path("/app/models/severity_classifier.pkl")
        self.severity_labels = {0: "Low", 1: "Medium", 2: "High"}
        self.model_version = "1.0.0"
        self.model_accuracy = 0.89
        
    def load_model(self):
        """Load the ML model"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                # Train model if it doesn't exist
                logger.warning("Model not found, training new model...")
                from train_model import train_and_save_model
                self.model = train_and_save_model()
                logger.info("New model trained and loaded")
            
            # Set model info metric
            ml_model_info.labels(
                model_name="LogisticRegression",
                version=self.model_version,
                accuracy=str(self.model_accuracy)
            ).set(1)
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> tuple:
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
        
        return prediction, confidence

# Initialize model container
model_container = ModelContainer()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting ML Inference API...")
    model_container.load_model()
    logger.info("ML Inference API ready")

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": "2.0.0",
        "stage": 2,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    model_loaded = model_container.model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "service": "ml-api",
        "model_loaded": model_loaded,
        "model_version": model_container.model_version,
        "stage": 2
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict ticket severity
    
    Returns severity classification with confidence score
    """
    start_time = time.time()
    
    try:
        # Validate model is loaded
        if model_container.model is None:
            ml_prediction_errors.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features
        features = np.array([[
            request.ticket_length,
            request.urgency_keywords,
            request.business_impact,
            request.customer_tier
        ]])
        
        # Make prediction
        prediction_code, confidence = model_container.predict(features)
        severity_label = model_container.severity_labels[prediction_code]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Update metrics
        ml_prediction_latency.observe(time.time() - start_time)
        ml_predictions_total.labels(
            severity=severity_label,
            status="success"
        ).inc()
        
        logger.info(
            f"Prediction: severity={severity_label}, "
            f"confidence={confidence:.2f}, time={processing_time:.2f}ms"
        )
        
        return PredictionResponse(
            severity=severity_label,
            severity_code=prediction_code,
            confidence=round(confidence, 4),
            processing_time_ms=round(processing_time, 2),
            model_version=model_container.model_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ml_prediction_errors.labels(error_type="prediction_error").inc()
        ml_predictions_total.labels(
            severity="unknown",
            status="error"
        ).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "LogisticRegression",
        "version": model_container.model_version,
        "accuracy": model_container.model_accuracy,
        "features": [
            "ticket_length",
            "urgency_keywords", 
            "business_impact",
            "customer_tier"
        ],
        "classes": model_container.severity_labels
    }

# ============================================================================
# STAGE 4: COMBINED ML + LLM PIPELINE
# ============================================================================

import httpx
from typing import Optional

# Additional Prometheus metrics for combined pipeline
triage_requests_total = Counter(
    'triage_requests_total',
    'Total triage requests',
    ['severity', 'status']
)

triage_latency = Histogram(
    'triage_latency_seconds',
    'End-to-end triage latency',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

triage_ml_llm_correlation = Gauge(
    'triage_ml_llm_correlation',
    'Correlation between ML confidence and LLM quality scores'
)

# Request/Response models for triage
class TriageRequest(BaseModel):
    ticket_text: str = Field(..., min_length=10, max_length=1000)
    ticket_length: int = Field(..., ge=0, le=1000)
    urgency_keywords: int = Field(..., ge=0, le=20)
    business_impact: float = Field(..., ge=0, le=10)
    customer_tier: int = Field(..., ge=1, le=5)

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_text": "URGENT: Production database is down affecting all customers",
                "ticket_length": 200,
                "urgency_keywords": 5,
                "business_impact": 10,
                "customer_tier": 5
            }
        }

class TriageResponse(BaseModel):
    severity: str
    severity_code: int
    confidence: float
    explanation: str
    recommended_actions: List[str]
    context_sources: List[str]
    processing_time_ms: float
    ml_processing_ms: float
    llm_processing_ms: float
    llm_quality_scores: Optional[Dict[str, float]] = None
    warnings: List[str] = []

async def call_llm_api(ticket_text: str, severity: str, timeout: float = 30.0) -> Dict:
    """
    Call LLM RAG API to get explanation for the ticket
    """
    llm_api_url = os.getenv("LLM_API_URL", "http://llm-rag-api:8001")
    
    # Construct query for LLM
    query = f"This is a {severity} severity ticket: {ticket_text}. What should be done?"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{llm_api_url}/chat",
                json={
                    "query": query,
                    "session_id": f"triage-{int(time.time())}",
                    "num_context_docs": 3
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        logger.error("LLM API timeout")
        raise HTTPException(status_code=504, detail="LLM API timeout")
    except httpx.HTTPError as e:
        logger.error(f"LLM API error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling LLM API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_recommended_actions(llm_response: str, severity: str) -> List[str]:
    """Extract actionable items from LLM response"""
    # Simple extraction - in production, use more sophisticated NLP
    actions = []
    
    if severity == "High":
        actions.append("Escalate to on-call engineer immediately")
        actions.append("Notify management")
        actions.append("Check system status dashboard")
    elif severity == "Medium":
        actions.append("Assign to appropriate team")
        actions.append("Investigate within 4 hours")
        actions.append("Update customer on progress")
    else:  # Low
        actions.append("Add to backlog")
        actions.append("Review in next sprint planning")
        actions.append("Send acknowledgment to customer")
    
    return actions

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """
    Combined ML + LLM pipeline for ticket triage
    
    Flow:
    1. ML model predicts severity
    2. LLM generates contextual explanation
    3. Return combined intelligent response
    """
    start_time = time.time()
    warnings = []
    llm_response_data = None
    llm_processing_time = 0.0
    
    try:
        # Step 1: ML Prediction
        ml_start = time.time()
        
        if model_container.model is None:
            raise HTTPException(status_code=503, detail="ML model not loaded")
        
        # Prepare features for ML model
        features = np.array([[
            request.ticket_length,
            request.urgency_keywords,
            request.business_impact,
            request.customer_tier
        ]])
        
        # Get ML prediction
        prediction_code, confidence = model_container.predict(features)
        severity_label = model_container.severity_labels[prediction_code]
        
        ml_processing_time = (time.time() - ml_start) * 1000  # ms
        
        logger.info(
            f"ML Prediction: severity={severity_label}, "
            f"confidence={confidence:.2f}, time={ml_processing_time:.2f}ms"
        )
        
        # Step 2: LLM Explanation
        llm_start = time.time()
        explanation = ""
        context_sources = []
        llm_quality_scores = {}
        
        try:
            llm_response_data = await call_llm_api(
                request.ticket_text,
                severity_label
            )
            
            explanation = llm_response_data.get("response", "")
            context_sources = [
                ctx[:100] + "..." if len(ctx) > 100 else ctx 
                for ctx in llm_response_data.get("context_used", [])
            ]
            
            # Extract LLM quality scores
            llm_quality_scores = {
                "relevance": llm_response_data.get("relevance_score", 0.0),
                "hallucination": llm_response_data.get("hallucination_score", 0.0),
                "groundedness": llm_response_data.get("groundedness_score", 0.0)
            }
            
            llm_processing_time = (time.time() - llm_start) * 1000  # ms
            
            logger.info(
                f"LLM Response: relevance={llm_quality_scores['relevance']:.2f}, "
                f"time={llm_processing_time:.2f}ms"
            )
            
        except HTTPException as e:
            # LLM API failed, use fallback
            warnings.append(f"LLM API unavailable: {e.detail}")
            explanation = f"Based on ML analysis, this is a {severity_label} severity issue. " \
                         f"Manual review recommended."
            context_sources = ["Fallback: ML prediction only"]
            llm_processing_time = (time.time() - llm_start) * 1000
            
            logger.warning(f"LLM API failed, using fallback: {e.detail}")
        
        # Step 3: Generate recommended actions
        recommended_actions = extract_recommended_actions(explanation, severity_label)
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000  # ms
        
        # Update metrics
        triage_latency.observe(time.time() - start_time)
        triage_requests_total.labels(
            severity=severity_label,
            status="success"
        ).inc()
        
        # Update correlation metric if LLM data available
        if llm_quality_scores:
            correlation = (confidence + llm_quality_scores["relevance"]) / 2.0
            triage_ml_llm_correlation.set(correlation)
        
        # Also update individual service metrics
        ml_predictions_total.labels(
            severity=severity_label,
            status="success"
        ).inc()
        
        logger.info(
            f"Triage complete: severity={severity_label}, "
            f"total_time={total_processing_time:.2f}ms"
        )
        
        return TriageResponse(
            severity=severity_label,
            severity_code=prediction_code,
            confidence=round(confidence, 4),
            explanation=explanation,
            recommended_actions=recommended_actions,
            context_sources=context_sources,
            processing_time_ms=round(total_processing_time, 2),
            ml_processing_ms=round(ml_processing_time, 2),
            llm_processing_ms=round(llm_processing_time, 2),
            llm_quality_scores=llm_quality_scores if llm_quality_scores else None,
            warnings=warnings
        )
        
    except HTTPException:
        triage_requests_total.labels(
            severity="unknown",
            status="error"
        ).inc()
        raise
    except Exception as e:
        triage_requests_total.labels(
            severity="unknown",
            status="error"
        ).inc()
        logger.error(f"Triage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STAGE 5: DRIFT DETECTION
# ============================================================================

from collections import deque
from datetime import datetime
from scipy import stats as scipy_stats

# Drift detection metrics
ml_data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift score (KL divergence)'
)

ml_drift_samples_analyzed = Gauge(
    'ml_drift_samples_analyzed',
    'Number of samples in drift analysis window'
)

ml_feature_drift = Gauge(
    'ml_feature_drift',
    'Per-feature drift score',
    ['feature_name']
)

class DriftDetector:
    """Synthetic drift detector using KL divergence"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reference_data = deque(maxlen=window_size)
        self.current_data = deque(maxlen=window_size)
        self.reference_set = False
        self.drift_threshold = 0.5
        
    def add_sample(self, features: np.ndarray):
        """Add a new sample for drift analysis"""
        if not self.reference_set:
            # Build reference distribution
            self.reference_data.append(features)
            if len(self.reference_data) >= self.window_size:
                self.reference_set = True
                logger.info("Reference distribution established for drift detection")
        else:
            # Monitor current distribution
            self.current_data.append(features)
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = np.asarray(p) + epsilon
        q = np.asarray(q) + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))
    
    def compute_drift_score(self) -> Dict:
        """Compute drift score between reference and current distributions"""
        if not self.reference_set or len(self.current_data) < 20:
            return {
                "drift_score": 0.0,
                "drift_detected": False,
                "samples_analyzed": len(self.current_data),
                "feature_drifts": {}
            }
        
        ref_array = np.array(list(self.reference_data))
        cur_array = np.array(list(self.current_data))
        
        # Calculate overall drift score (average across features)
        feature_drifts = {}
        drift_scores = []
        
        feature_names = ["ticket_length", "urgency_keywords", "business_impact", "customer_tier"]
        
        for i, feature_name in enumerate(feature_names):
            ref_feature = ref_array[:, i]
            cur_feature = cur_array[:, i]
            
            # Create histograms
            bins = 10
            ref_hist, _ = np.histogram(ref_feature, bins=bins, density=True)
            cur_hist, _ = np.histogram(cur_feature, bins=bins, density=True)
            
            # Calculate KL divergence
            kl_div = self.calculate_kl_divergence(ref_hist, cur_hist)
            drift_scores.append(kl_div)
            feature_drifts[feature_name] = float(kl_div)
            
            # Update per-feature metric
            ml_feature_drift.labels(feature_name=feature_name).set(kl_div)
        
        overall_drift = np.mean(drift_scores)
        drift_detected = overall_drift > self.drift_threshold
        
        # Update metrics
        ml_data_drift_score.set(overall_drift)
        ml_drift_samples_analyzed.set(len(self.current_data))
        
        return {
            "drift_score": round(float(overall_drift), 4),
            "drift_detected": drift_detected,
            "threshold": self.drift_threshold,
            "samples_analyzed": len(self.current_data),
            "feature_drifts": feature_drifts,
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }

# Initialize drift detector
drift_detector = DriftDetector(window_size=100)

# Modify predict endpoint to track drift
@app.post("/predict_v2", response_model=PredictionResponse)
async def predict_with_drift_tracking(request: PredictionRequest):
    """
    Predict endpoint with drift tracking (Stage 5)
    """
    # Add to drift detector
    features = np.array([[
        request.ticket_length,
        request.urgency_keywords,
        request.business_impact,
        request.customer_tier
    ]])
    drift_detector.add_sample(features[0])
    
    # Call original predict logic
    return await predict(request)

@app.get("/drift/status")
async def drift_status():
    """Get current drift detection status"""
    return drift_detector.compute_drift_score()

@app.post("/drift/reset")
async def reset_drift_detector():
    """Reset drift detector (for testing)"""
    global drift_detector
    drift_detector = DriftDetector(window_size=100)
    return {"status": "reset", "message": "Drift detector reset successfully"}
