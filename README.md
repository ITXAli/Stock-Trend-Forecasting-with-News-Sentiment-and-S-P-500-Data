# 📈 Stock Market News-Based Trend Predictor

This project predicts whether the stock market (specifically the S&P 500 index) is likely to trend **Up** or **Down** based on real-world news and historical stock indicators. It combines **natural language sentiment analysis** using (NLP VADER) with **quantitative market features**, powered by a machine learning model trained using `XGBoost`.

---

## 🚀 Features

- 📰 Takes raw news headlines or articles as input
- 📊 Allows manual input of stock indicators (open, high, low, close, volume, etc.)
- 🤖 Uses a trained `XGBoost` classification model to predict trend direction
- 🌐 Includes a responsive frontend (`index.html`)
- 🔥 Built with `Flask` backend + `VADER` for sentiment scoring

---

## 📂 Folder Structure

  ├── app.py # Flask backend (serves predictions)
  
  ├── best_xgb_model.joblib # Trained XGBoost model
  
  ├── index.html # Frontend for input and visualization
  
  ├── deployment.ipynb # Model training & saving
  
  ├── Feature_Engineering.ipynb
  
  ├── Extra_Features.ipynb # Additional feature calculations
  
  ├── stock(updated).xlsx # Final cleaned dataset used for training
  
  ├── merged_data.xlsx # Combined data from multiple sources
  
  ├── project.ipynb # Final integrated pipeline notebook
  
  ├── requirement.txt # Dependencies list


---

## 🧠 How It Works

1. **Frontend** accepts:
   - News text
   - Stock metrics (open, close, volume, etc.)

2. **Backend (`app.py`)**:
   - Uses VADER to calculate sentiment
   - Constructs a 22-feature vector
   - Feeds it into the saved XGBoost model
   - Returns trend prediction and probabilities

---

## 🔧 Installation

1. Clone the repo:
   
   git clone https://github.com/yourusername/stock-trend-predictor.git
   
   cd stock-trend-predictor
   
3. Create a virtual environment and install dependencies:

    python -m venv venv
   
    venv\Scripts\activate  # Windows
   
    pip install -r requirement.
    
   
3. Run the Flask app:

    python app.py
  
4. Open index.html in your browser to test the interface.

## 📌 Sample Test Input (for API / Postman)

{
  "news_text": "Apple reports record revenue for Q2 as iPhone sales surge beyond expectations.",
  
  "Close_^GSPC": 5300.25,
  
  "High_^GSPC": 5320.00,
  
  "Low_^GSPC": 5280.10,
  
  "Open_^GSPC": 5290.00,
  
  "Volume_^GSPC": 4000000000,
  
  "daily_return": 0.0125,
  
  "Close_lag_1": 5235.00,
  
  "Close_lag_2": 5200.00,
  
  "Close_lag_3": 5185.00,
  
  "Close_MA_5": 5210.80,
  
  "Volume_MA_5": 3900000000,
  
  "Close_Std_5": 18.75
  
}
## 🛠 Tech Stack

  Python (Flask, NumPy, pandas, joblib, scikit-learn, xgboost)
  
  HTML5, JavaScript (Fetch API)
  
  VADER Sentiment Analysis (NLTK)
  
  Excel + Jupyter notebooks for data preprocessing

## 📊 Model Performance

  Accuracy: ~78%
  
  Baseline Accuracy: 52%
  
  ROC AUC: ~0.85
  
  Evaluated using TimeSeriesSplit and engineered 

## 📌 Notes

  For live stock integration, you can extend this using yfinance.
  
  To switch to regression for predicting percent change instead of direction, consider XGBRegressor

## 📃 License

This project is for academic and educational use.

