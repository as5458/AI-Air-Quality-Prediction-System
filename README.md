# 🌍 AI-Powered Air Quality Forecasting & Analysis

An AI-powered web application that predicts and analyzes Air Quality Index (AQI) using Machine Learning models and real-time environmental APIs.  
The system provides current AQI status, 5-day AQI forecasting, pollutant concentration analysis, and interactive visualizations.

---

## 🚀 Features

- 🔍 Search AQI by city
- 📊 Real-time AQI monitoring
- 🤖 Machine Learning based AQI prediction
- 📈 5-Day AQI Forecast Graph
- 🌫️ Pollutant concentration analysis
- 📉 Interactive charts and analytics
- 🌐 Integration with OpenWeather API & WAQI API
- 💻 Responsive frontend dashboard

---

## 🛠️ Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

### Backend
- Flask (Python)

### Machine Learning
- Linear Regression
- Random Forest Regressor
- Support Vector Regressor (SVR)

### APIs
- OpenWeather API
- WAQI API

---

## 📂 Project Structure

```bash
AI-Air-Quality-Prediction-System/
│
├── dataset/
│   └── aqi_dataset.csv
│
├── frontend/
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── app.py
│   └── cities.txt
│
├── models/
│   ├── train_models.py
│   ├── check_accuracy.py
│
├── .gitignore
└── README.md
