import config

class CycloneRiskAssessor:
    def __init__(self):
        self.thresholds = getattr(config, 'RISK_THRESHOLDS', {})
        self.demo_mode = getattr(config, 'DEMO_MODE', True)

    def _estimate_coastal_proximity(self, lat, lon):
        if lat < 8 or lat > 22 or lon < 68 or lon > 90:
            return 0.1
            
        dist_to_coast = abs(lon - 80) + abs(lat - 15)
        score = max(0, 1.0 - (dist_to_coast / 15.0))
        return min(1.0, score)

    def _estimate_intensity_trend(self, track):
        if not track or len(track) < 2:
            return "stable"
            
        winds = [p.get('wind', p.get('wind_estimated', 0)) for p in track]
        if len(winds) >= 2:
            if winds[-1] > winds[0] + 10:
                return "increasing"
            elif winds[-1] < winds[0] - 10:
                return "decreasing"
        return "stable"

    def assess_risk(self, wind_speed, predicted_track, current_position, pressure=None):
        try:
            wind_score = min(100, max(0, (wind_speed - 30) / 100 * 100))
            
            curr_lat = current_position.get('lat', 15)
            curr_lon = current_position.get('lon', 85)
            prox_score = self._estimate_coastal_proximity(curr_lat, curr_lon) * 100
            
            trend = self._estimate_intensity_trend(predicted_track)
            trend_score = 100 if trend == "increasing" else (50 if trend == "stable" else 10)
            
            pressure_score = 50
            if pressure:
                pressure_score = min(100, max(0, (1010 - pressure) / 100 * 100))
                
            risk_score = (wind_score * 0.4) + (prox_score * 0.3) + (trend_score * 0.2) + (pressure_score * 0.1)
            risk_score = round(risk_score, 1)
            
            if risk_score < 25:
                risk_level = "LOW"
            elif risk_score < 50:
                risk_level = "MODERATE"
            elif risk_score < 75:
                risk_level = "HIGH"
            else:
                risk_level = "VERY HIGH"
                
            reason = f"Risk is {risk_level} primarily due to wind speeds of {wind_speed} knots and a {trend} intensity trend."
            
            recommended_actions = []
            if risk_level in ["HIGH", "VERY HIGH"]:
                recommended_actions = ["Monitor official weather authorities", "Secure loose objects if safe", "Follow verified local guidance"]
            elif risk_level == "MODERATE":
                recommended_actions = ["Monitor official weather updates", "Prepare emergency supplies", "Follow verified local guidance"]
            else:
                recommended_actions = ["Stay informed through official channels"]
                
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "reason": reason,
                "recommended_actions": recommended_actions,
                "factors": {
                    "wind_score": round(wind_score, 1),
                    "proximity_score": round(prox_score, 1),
                    "trend_score": round(trend_score, 1),
                    "pressure_score": round(pressure_score, 1)
                },
                "demo_mode": self.demo_mode
            }
        except Exception as e:
            print(f"Error assessing risk: {e}")
            return {
                "risk_level": "UNKNOWN",
                "risk_score": 0,
                "reason": f"Error calculating risk: {str(e)}",
                "recommended_actions": [],
                "factors": {},
                "demo_mode": True
            }
