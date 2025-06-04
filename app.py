from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)

# Load your trained XGBClassifier model
model = joblib.load('best_xgb_model.joblib')

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('news_text', '')

        if not text.strip():
            return jsonify({'error': 'Empty news text provided.'}), 400

        # Required stock input fields
        required_fields = [
            'Close_^GSPC', 'High_^GSPC', 'Low_^GSPC', 'Open_^GSPC', 'Volume_^GSPC',
            'daily_return', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3',
            'Close_MA_5', 'Volume_MA_5', 'Close_Std_5'
        ]

        # Validate input
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required stock field: {field}'}), 400

        # Text-based features
        sentiment_score = vader.polarity_scores(text)['compound']
        text_length = len(text)
        word_count = len(text.split())
        now = datetime.datetime.now()
        day_of_week = now.weekday()
        month = now.month

        # Stock-based features from frontend
        close = float(data['Close_^GSPC'])
        high = float(data['High_^GSPC'])
        low = float(data['Low_^GSPC'])
        open_ = float(data['Open_^GSPC'])
        volume = float(data['Volume_^GSPC'])
        price_range = high - low
        daily_return = float(data['daily_return'])
        close_lags = [float(data['Close_lag_1']), float(data['Close_lag_2']), float(data['Close_lag_3'])]
        close_ma_5 = float(data['Close_MA_5'])
        volume_ma_5 = float(data['Volume_MA_5'])
        close_std_5 = float(data['Close_Std_5'])

        # Feature vector (must match training order)
        feature_vector = np.array([
            sentiment_score,
            text_length,
            word_count,
            day_of_week,
            month,
            close,
            high,
            low,
            open_,
            volume,
            price_range,
            daily_return,
            close_lags[0],
            close_lags[1],
            close_lags[2],
            sentiment_score,  # lag_1 placeholder
            sentiment_score,  # lag_2 placeholder
            sentiment_score,  # lag_3 placeholder
            close_ma_5,
            volume_ma_5,
            sentiment_score,  # sentiment_MA_5 placeholder
            close_std_5
        ], dtype=np.float64).reshape(1, -1)

        # Check feature validity
        if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
            return jsonify({'error': 'Feature vector contains NaN or Inf values.'}), 400

        # Prediction
        prediction = model.predict(feature_vector)[0]
        proba = model.predict_proba(feature_vector)[0]
        classes = model.classes_

        # Map numeric prediction to label
        label_map = {0: "Down", 1: "Up"}
        predicted_label = label_map.get(int(prediction), "Unknown")

        return jsonify({
            'prediction': predicted_label,
            'probabilities': {
                label_map.get(int(classes[0]), str(classes[0])): round(float(proba[0]), 4),
                label_map.get(int(classes[1]), str(classes[1])): round(float(proba[1]), 4)
            }
        })


    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
