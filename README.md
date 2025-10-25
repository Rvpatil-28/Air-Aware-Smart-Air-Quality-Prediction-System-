🌍 AirAware — Air Quality Prediction System (Demo)

Predict • Visualize • Act — A modern, interactive tool to monitor and forecast air quality using time-series analysis and intuitive visualizations.

✨ Overview

AirAware is a demo project that demonstrates how data science and forecasting techniques can be applied to environmental monitoring.
It predicts air quality levels (PM2.5, PM10, NO₂, etc.) using Prophet and visualizes them through a sleek Streamlit dashboard.

🧩 Features

🧹 Data Preprocessing Utilities – Clean and prepare air quality datasets for modeling.

📈 Time-Series Forecasting (Prophet) – Predict pollutant levels based on historical data.

🖥️ Interactive Streamlit Dashboard – Visualize trends, forecasts, and alerts in real time.

🧪 Synthetic Demo Dataset – Comes with an example file: data/sample_aq.csv.

🌐 OpenAQ Data Integration – Fetch real-world data via the OpenAQ API.

📧 Email Alert System (Optional) – Send alerts when pollutant levels exceed thresholds.

🚀 Quick Start
1️⃣ Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (Optional for Demo)
python src/train_model.py --csv data/sample_aq.csv --target pm25 --model_out models/prophet_pm25.joblib

4️⃣ Run the Dashboard
streamlit run app.py


💡 Open the URL shown in your terminal to access the dashboard (usually http://localhost:8501).

📊 Using Your Own Data

To use a custom dataset:

Include a date/time column named date or ds.

Include pollutant columns like pm25, pm10, no2, etc.

Example CSV header:

date,pm25,pm10,no2
2024-01-01,34,65,22
2024-01-02,28,58,19


The preprocessing and model scripts will automatically adapt.
Check out src/utils.py
 for customization options.

🌐 Fetching Real Data from OpenAQ

To fetch live air quality data:

python src/fetch_openaq.py --city "Delhi" --pollutant pm25


Or, enable the OpenAQ data option in the Streamlit sidebar to visualize real-time updates.

📧 Email Alerts (Optional)

To activate the alert system:

Configure your SMTP settings inside src/alerts.py
.

Use the function to send notifications when pollutant thresholds are exceeded.

⚠️ This is a demo feature. Update your SMTP credentials carefully!

📁 Project Structure
AirAware/
├── app.py                  # Streamlit dashboard
├── data/
│   └── sample_aq.csv       # Example dataset
├── models/                 # Saved Prophet models
├── src/
│   ├── utils.py            # Data preprocessing helpers
│   ├── train_model.py      # Training pipeline
│   ├── fetch_openaq.py     # OpenAQ data fetcher
│   └── alerts.py           # Email alert demo
└── requirements.txt        # Dependencies

🧠 Tech Stack
Component	Technology
Language	Python 3.x
Forecasting Model	Facebook Prophet
Dashboard	Streamlit
Data Source	OpenAQ API
Alerts	SMTP (Email)
🤝 Contributing

Contributions, bug reports, and suggestions are welcome!
Feel free to fork the repo and submit a pull request.

📜 License

This project is released under the MIT License — free to use, modify, and distribute.

🧑‍💻 Author

Ritesh Patil
💬 Data Science Enthusiast | AI & ML Learner
📧 [your.email@example.com
]
🌐 GitHub Profile

“Clean air is everyone’s right. Let’s predict and protect!” 🌱
