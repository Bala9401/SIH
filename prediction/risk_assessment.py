import math
import config

class CycloneRiskAssessor:
    def __init__(self):
        self.thresholds = getattr(config, 'RISK_THRESHOLDS', {})
        self.demo_mode = getattr(config, 'DEMO_MODE', True)

    def _estimate_coastal_proximity(self, lat, lon):
        # Prototype coastline reference points for India's east and west coasts.
        coastline = [(8.1, 77.5), (10.0, 76.2), (15.0, 73.8), (19.0, 72.8),
                     (21.5, 69.5), (20.3, 86.7), (16.5, 82.3), (13.0, 80.3),
                     (10.8, 79.8), (22.0, 88.0)]
        def distance_km(a, b):
            lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            return 6371.0 * 2 * math.asin(math.sqrt(hav))
        distance = min(distance_km((lat, lon), point) for point in coastline)
        return max(0.0, min(1.0, 1.0 - distance / 500.0)), distance

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
            proximity, distance_to_coast = self._estimate_coastal_proximity(curr_lat, curr_lon)
            prox_score = proximity * 100
            
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
                    ,"distance_to_coast_km": round(distance_to_coast, 1)
                },
                "distance_to_coast_km": round(distance_to_coast, 1),
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
