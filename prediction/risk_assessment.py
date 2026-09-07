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

    def _coastal_score_for_track(self, track):
        scores = []
        distances = []
        for point in track:
            if point.get('lat') is None or point.get('lon') is None:
                continue
            score, distance = self._estimate_coastal_proximity(point['lat'], point['lon'])
            scores.append(score * 100)
            distances.append(distance)
        return (max(scores) if scores else 0.0), (min(distances) if distances else None)

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

    def _risk_level(self, score):
        if score < 25:
            return "LOW"
        if score < 50:
            return "MODERATE"
        if score < 75:
            return "HIGH"
        return "VERY HIGH"

    def assess_risk(self, wind_speed, predicted_track, current_position, pressure=None,
                    satellite_analysis=None):
        try:
            if wind_speed is None or not current_position:
                return {
                    "risk_level": "INSUFFICIENT DATA",
                    "risk_score": None,
                    "reason": "Current wind and cyclone position are required for meteorological risk calculation.",
                    "recommended_actions": ["Obtain verified meteorological observations and follow official advisories."],
                    "satellite_analysis": satellite_analysis,
                    "meteorological_factors": {"wind_speed": wind_speed, "pressure": pressure},
                    "factors": {},
                    "risk_weights": {},
                    "distance_to_coast_km": None,
                    "risk_source": "meteorological_track_and_coastal_analysis",
                    "satellite_contribution": "unavailable",
                    "final_risk_assessment": "Insufficient meteorological data.",
                    "demo_mode": self.demo_mode
                }
            wind_score = min(100, max(0, (float(wind_speed) - 20) / 80 * 100))
            
            curr_lat = current_position.get('lat', 15)
            curr_lon = current_position.get('lon', 85)
            proximity, distance_to_coast = self._estimate_coastal_proximity(curr_lat, curr_lon)
            current_coast_score = proximity * 100
            forecast_coast_score, forecast_distance = self._coastal_score_for_track(predicted_track)
            prox_score = (current_coast_score * 0.4 + forecast_coast_score * 0.6
                          if predicted_track else current_coast_score)
            
            trend = self._estimate_intensity_trend(predicted_track)
            trend_score = 100 if trend == "increasing" else (50 if trend == "stable" else 0)
            
            pressure_score = None
            if pressure is not None:
                pressure_score = min(100, max(0, (1010 - float(pressure)) / 60 * 100))

            uncertainty_values = [p.get('uncertainty_radius_km') for p in predicted_track
                                  if p.get('uncertainty_radius_km') is not None]
            uncertainty_score = min(100, max(0, (max(uncertainty_values) if uncertainty_values else 0) / 300 * 100))
            image_score = None
            if satellite_analysis and satellite_analysis.get('image_risk_score') is not None:
                image_score = min(100, max(0, float(satellite_analysis['image_risk_score'])))

            weights = {'satellite': 0.30, 'wind': 0.25, 'pressure': 0.15,
                       'proximity': 0.15, 'trend': 0.10, 'uncertainty': 0.05}
            values = {'satellite': image_score, 'wind': wind_score,
                      'pressure': pressure_score, 'proximity': prox_score,
                      'trend': trend_score, 'uncertainty': uncertainty_score}
            available = {name: value for name, value in values.items() if value is not None}
            weight_total = sum(weights[name] for name in available)
            risk_score = sum(available[name] * weights[name] for name in available) / weight_total
            risk_score = round(risk_score, 1)

            risk_level = self._risk_level(risk_score)
                
            reason_parts = [f"wind {float(wind_speed):.1f} km/h",
                            f"{trend} forecast trend",
                            f"nearest forecast coast distance {forecast_distance:.1f} km" if forecast_distance is not None else "no forecast coast approach"]
            if pressure is None:
                reason_parts.append("pressure unavailable; weights redistributed")
            if image_score is not None:
                reason_parts.append(f"satellite product context score {image_score:.1f}/100")
            else:
                reason_parts.append("satellite image unavailable; weights redistributed")
            reason = f"Risk is {risk_level}, based on " + ", ".join(reason_parts) + "."
            
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
                "satellite_analysis": satellite_analysis,
                "image_risk_score": image_score,
                "risk_contribution": {
                    key: round(available[key] * weights[key] / weight_total, 1)
                    if key in available else None
                    for key in weights
                },
                "meteorological_factors": {
                    "wind_speed": wind_speed,
                    "pressure": pressure,
                    "distance_to_coast_km": round(distance_to_coast, 1),
                    "coastal_distance": round(distance_to_coast, 1),
                    "forecast_distance_to_coast_km": round(forecast_distance, 1) if forecast_distance is not None else None,
                    "trend": trend,
                    "intensity_trend": trend,
                    "uncertainty_available": bool(uncertainty_values)
                },
                "factors": {f"{name}_score": (round(value, 1) if value is not None else None)
                            for name, value in values.items()},
                "distance_to_coast_km": round(distance_to_coast, 1),
                "risk_weights": {name: round(weights[name] / weight_total, 3) for name in available},
                "final_risk_assessment": "Transparent weighted assessment; satellite product class contributes bounded contextual evidence only.",
                "risk_source": "satellite_context_and_meteorological_track_analysis",
                "satellite_contribution": "bounded_contextual_score" if image_score is not None else "unavailable",
                "demo_mode": self.demo_mode
            }
        except Exception as e:
            print(f"Error assessing risk: {e}")
            return {
                "risk_level": "INSUFFICIENT DATA",
                "risk_score": None,
                "reason": f"Error calculating risk: {str(e)}",
                "recommended_actions": [],
                "satellite_analysis": satellite_analysis,
                "meteorological_factors": {},
                "factors": {},
                "risk_weights": {},
                "distance_to_coast_km": None,
                "risk_source": "meteorological_track_and_coastal_analysis",
                    "satellite_contribution": "unavailable",
                "demo_mode": True
            }
