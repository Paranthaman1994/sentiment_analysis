from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pre-trained model and vectorizer
model_rf = joblib.load(os.path.join('models', 'svm_final_model.pkl'))
vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer_final_model.pkl'))

# Define route for inference
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        input_text = data.get('text')

        if not input_text:
            return jsonify({'error': 'No text provided'}), 400

        # Vectorize input text
        vectorized_text = vectorizer.transform([input_text])

        # Make prediction
        prediction = model_rf.predict(vectorized_text)

        # Convert prediction to standard Python int
        prediction_value = int(prediction[0])  # Assuming it's a single output

        # Map prediction value to sentiment
        sentiment = "positive" if prediction_value == 1 else "negative"

        # Return the result as JSON
        response = {
            'input_text': input_text,
            'prediction': prediction_value,
            'sentiment': sentiment
        }
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
