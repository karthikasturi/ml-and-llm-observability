"""
Train ML Model for Ticket Severity Classification
Generates synthetic training data and trains a Logistic Regression model
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic customer support ticket data
    
    Features:
    - ticket_length: number of words (proxy for detail)
    - urgency_keywords: count of urgent terms
    - business_impact: score 0-10
    - customer_tier: 1-5 (VIP = 5)
    
    Target:
    - severity: 0 (Low), 1 (Medium), 2 (High)
    """
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Generate base features
        ticket_length = np.random.randint(20, 300)
        urgency_keywords = np.random.randint(0, 10)
        business_impact = np.random.uniform(0, 10)
        customer_tier = np.random.randint(1, 6)
        
        # Logic to determine severity
        # High severity: high impact + urgency + VIP customer
        if business_impact > 7 and urgency_keywords > 4 and customer_tier >= 4:
            severity = 2  # High
        # Low severity: low impact + no urgency
        elif business_impact < 4 and urgency_keywords < 2:
            severity = 0  # Low
        # Medium severity: everything else with some noise
        else:
            # Add some randomness
            if np.random.random() < 0.2:
                severity = np.random.choice([0, 2])
            else:
                severity = 1  # Medium
        
        data.append({
            'ticket_length': ticket_length,
            'urgency_keywords': urgency_keywords,
            'business_impact': business_impact,
            'customer_tier': customer_tier,
            'severity': severity
        })
    
    return pd.DataFrame(data)

def train_and_save_model(save_path: str = "/app/models/severity_classifier.pkl"):
    """Train and save the model"""
    logger.info("Generating synthetic training data...")
    df = generate_synthetic_data(n_samples=2000)
    
    # Split features and target
    X = df[['ticket_length', 'urgency_keywords', 'business_impact', 'customer_tier']]
    y = df['severity']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Class distribution:\n{y.value_counts()}")
    
    # Train model
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(
        y_test, y_pred,
        target_names=['Low', 'Medium', 'High']
    ))
    
    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info(f"Model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    train_and_save_model()
