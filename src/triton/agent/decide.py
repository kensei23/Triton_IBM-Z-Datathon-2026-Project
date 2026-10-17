from configs import thresholds


def decide(event):
    """Turn a pollution event into a recommendation using fixed rules."""
    prob = event["pollution_probability"]
    eta = event["eta_minutes_to_intake"]

    close_intake = (
        prob >= thresholds["close_prob"] and eta < thresholds["close_eta_min"]
    )
    alert_ea = prob >= thresholds["alert_prob"]

    if close_intake:
        severity = "high"
    elif alert_ea:
        severity = "medium"
    else:
        severity = "low"

    return {
        "close_intake": close_intake,
        "alert_ea": alert_ea,
        "severity": severity,
        "probability_pct": round(prob * 100),
        "eta_minutes": eta,
        "station_id": event["station_id"],
        "intake_id": event["intake_id"],
    }
