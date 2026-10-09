"""
Confidence Calibration & Uncertainty Estimation for Medical Deep Learning Models.
Calculates prediction entropy, probability margins, temperature scaling calibration,
and flags low-confidence predictions requiring urgent clinical review.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import UNCERTAINTY_THRESHOLD_LOW, UNCERTAINTY_THRESHOLD_HIGH, DISEASE_CLASSES


def compute_temperature_scaling(probabilities, temperature=1.2):
    """
    Applies temperature scaling calibration to smooth overconfident neural net outputs.
    """
    logits = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / (1 - np.clip(probabilities, 1e-7, 1 - 1e-7)))
    calibrated_logits = logits / temperature
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    return calibrated_probs


# Alias for API compatibility
compute_confidence_calibration = compute_temperature_scaling



def estimate_uncertainty(probabilities_dict):
    """
    Analyzes multi-label probabilities to compute:
    1. Maximum Finding Confidence
    2. Multi-label Predictive Entropy
    3. Probability Margin
    4. Screening Risk Level & Confidence Rating (High, Moderate, Low/Uncertain)
    5. Warning flag if prediction falls in low-confidence / high-ambiguity zone.
    """
    probs = np.array(list(probabilities_dict.values()))

    # 1. Primary Finding & Top Probability
    sorted_findings = sorted(probabilities_dict.items(), key=lambda x: x[1], reverse=True)
    top_finding, top_prob = sorted_findings[0]
    second_finding, second_prob = sorted_findings[1] if len(sorted_findings) > 1 else ("None", 0.0)

    # 2. Predictive Binary Entropy (bits)
    # Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
    clipped_probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    entropies = -(clipped_probs * np.log2(clipped_probs) + (1 - clipped_probs) * np.log2(1 - clipped_probs))
    mean_entropy = float(np.mean(entropies))

    # 3. Probability Margin (top_prob - second_prob)
    margin = float(top_prob - second_prob)

    # 4. Determine Confidence Category & Flag
    if top_prob >= UNCERTAINTY_THRESHOLD_HIGH and mean_entropy < 0.45:
        confidence_rating = "HIGH"
        confidence_color = "#10b981" # Green
        low_confidence_flag = False
        warning_msg = None
    elif top_prob >= UNCERTAINTY_THRESHOLD_LOW:
        confidence_rating = "MODERATE"
        confidence_color = "#f59e0b" # Yellow
        low_confidence_flag = False
        warning_msg = None
    else:
        confidence_rating = "LOW (UNCERTAIN)"
        confidence_color = "#ef4444" # Red
        low_confidence_flag = True
        warning_msg = "⚠️ Low-confidence result — high model ambiguity detected. Further clinical evaluation recommended."

    # Formulate output dictionary
    return {
        "primary_finding": top_finding,
        "primary_confidence_pct": round(top_prob * 100, 1),
        "primary_prob": top_prob,
        "second_finding": second_finding,
        "second_prob": second_prob,
        "confidence_rating": confidence_rating,
        "confidence_color": confidence_color,
        "predictive_entropy": round(mean_entropy, 3),
        "probability_margin": round(margin, 3),
        "low_confidence_flag": low_confidence_flag,
        "warning_message": warning_msg
    }
