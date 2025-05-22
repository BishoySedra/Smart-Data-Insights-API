# 🚀 Smart Data Insights API

A powerful and modular **FastAPI** service that enables:
- ✨ Intelligent sentiment analysis with sarcasm detection
- 📊 Auto data cleaning and profiling
- 📈 Forecasting using Prophet
- 🎯 Visual insights for business and education datasets
- ☁️ Seamless Cloudinary integration
- 🌐 Public access via Ngrok

---

## 📦 Features

- **Upload a CSV file** from Cloudinary and get:
  - Cleaned dataset (AutoClean)
  - Visual and statistical insights (matplotlib, seaborn)
  - Forecasts (Prophet)
  - Smart reports (business or student data)
- **Sentiment prediction API** for reviews with sarcasm-aware logic

---

## ⚙️ Technologies

- **Backend:** FastAPI + Uvicorn
- **Data Analysis:** pandas, seaborn, matplotlib, Prophet
- **ML & NLP:** scikit-learn, NLTK
- **Deployment:** Docker + Ngrok
- **Storage:** Cloudinary

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create `.env` file
```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
NGROK_AUTH_TOKEN=your_ngrok_token
```

### 3. Build and run the Docker container
```bash
docker build -t smart-api .
docker run -p 8000:8000 smart-api
```

---

## 📡 API Endpoints

### `/clean-data`  
**POST** – Cleans a CSV from Cloudinary

### `/analyze-data`  
**POST** – Generates insights & visualizations

### `/predict-review`  
**POST** – Returns sentiment (with sarcasm detection)

---

## 📁 Sample Request

```json
POST /clean-data
{
  "cloudinary_url": "https://res.cloudinary.com/...csv"
}
```
