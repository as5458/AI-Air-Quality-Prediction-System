from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WAQI_TOKEN = os.getenv("WAQI_TOKEN")

app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load cities
CITY_FILE = os.path.join(BASE_DIR, "cities.txt")
try:
    with open(CITY_FILE, "r", encoding="utf-8") as file:
        VALID_CITIES = {city.strip().lower() for city in file if city.strip()}
    print(f"Loaded {len(VALID_CITIES)} cities")
except Exception as e:
    print("Failed to load cities.txt:", e)
    VALID_CITIES = set()

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def pm25_to_aqi(pm25):
    if pm25 <= 12.0: return int((50 / 12.0) * pm25)
    elif pm25 <= 35.4: return int(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4: return int(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4: return int(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4: return int(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
    else: return 500

def get_pollutant_value(iaqi_data, pollutant):
    if pollutant in iaqi_data:
        val = iaqi_data[pollutant].get("v")
        return float(val) if val is not None else None
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        city = data.get("city", "").strip().lower()
        city = " ".join(city.split())
        print(f"User entered city: {city}")

        if not city:
            return jsonify({"error": "City required"}), 400
        if city not in VALID_CITIES:
            return jsonify({"error": f"{city} is not a valid city"}), 404

        # ============================================
        # STEP 1 - GET COORDINATES FIRST (OpenWeather)
        # ============================================
        geo_url = (
            "https://api.openweathermap.org/geo/1.0/direct"
            f"?q={city}&limit=1"
            f"&appid={OPENWEATHER_API_KEY}"
        )
        geo_res = requests.get(geo_url, timeout=10).json()

        if not geo_res:
            return jsonify({"error": "Coordinates not found"}), 404

        lat = geo_res[0]["lat"]
        lon = geo_res[0]["lon"]
        print(f"Coordinates for {city}: {lat}, {lon}")

        # ============================================
        # STEP 2 - WAQI LIVE AQI (GEO-BASED - NEAREST STATION)
        # ============================================
        waqi_url = (
            f"https://api.waqi.info/feed/geo:{lat};{lon}/"
            f"?token={WAQI_TOKEN}"
        )
        waqi_res = requests.get(waqi_url, timeout=10).json()

        if waqi_res.get("status") != "ok":
            # Fallback: try city name endpoint
            encoded_city = quote(city)
            waqi_url_fallback = (
                f"https://api.waqi.info/feed/{encoded_city}/"
                f"?token={WAQI_TOKEN}"
            )
            waqi_res = requests.get(waqi_url_fallback, timeout=10).json()
            
            if waqi_res.get("status") != "ok":
                return jsonify({"error": "WAQI data not available"}), 404

        waqi_data = waqi_res["data"]

        # Extract AQI
        aqi_raw = waqi_data.get("aqi")
        if isinstance(aqi_raw, dict):
            current_aqi = int(aqi_raw.get("v", 0))
        else:
            current_aqi = int(aqi_raw) if aqi_raw is not None else 0

        dominant_pollutant = waqi_data.get("dominentpol", "N/A")
        iaqi = waqi_data.get("iaqi", {})

        live_pm25 = get_pollutant_value(iaqi, "pm25")
        live_pm10 = get_pollutant_value(iaqi, "pm10")
        live_no2 = get_pollutant_value(iaqi, "no2")
        live_co = get_pollutant_value(iaqi, "co")
        live_o3 = get_pollutant_value(iaqi, "o3")

        category = get_aqi_category(current_aqi)

        print(f"WAQI Live AQI for {city}: {current_aqi}")
        print(f"Station: {waqi_data.get('city', {}).get('name', 'Unknown')}")
        print(f"PM2.5: {live_pm25}, PM10: {live_pm10}")

        # ============================================
        # STEP 3 - OPENWEATHER FORECAST
        # ============================================
        forecast_url = (
            "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
            f"?lat={lat}&lon={lon}"
            f"&appid={OPENWEATHER_API_KEY}"
        )
        forecast_res = requests.get(forecast_url, timeout=10).json()
        forecast_list = forecast_res.get("list", [])

        if not forecast_list:
            return jsonify({"error": "Forecast data unavailable"}), 500

        daily_pm25 = []
        future_components = {"pm2_5": [], "pm10": [], "no2": [], "co": [], "o3": []}

        for i in range(0, min(40, len(forecast_list)), 8):
            chunk = forecast_list[i:i + 8]
            avg_pm25 = sum(item["components"]["pm2_5"] for item in chunk) / len(chunk)
            daily_pm25.append(avg_pm25)

            future_components["pm2_5"].append(round(avg_pm25, 2))
            future_components["pm10"].append(round(sum(item["components"]["pm10"] for item in chunk) / len(chunk), 2))
            future_components["no2"].append(round(sum(item["components"]["no2"] for item in chunk) / len(chunk), 2))
            future_components["co"].append(round(sum(item["components"]["co"] for item in chunk) / len(chunk), 2))
            future_components["o3"].append(round(sum(item["components"]["o3"] for item in chunk) / len(chunk), 2))

        predicted_values = [pm25_to_aqi(pm) for pm in daily_pm25]
        dates = [f"Day {i + 1}" for i in range(len(predicted_values))]

        return jsonify({
            "city": city.title(),
            "current_aqi": current_aqi,
            "category": category,
            "dominant_pollutant": dominant_pollutant,
            "live_pm25": live_pm25,
            "live_pm10": live_pm10,
            "live_no2": live_no2,
            "live_co": live_co,
            "live_o3": live_o3,
            "forecast": predicted_values,
            "dates": dates,
            "future_components": future_components
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "API request timeout"}), 500
    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)