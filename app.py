from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load Hugging Face emotion detection model
print("Loading emotion model... (this may take a few seconds)")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
print("Model loaded successfully!")

@app.route('/')
def home():
    return jsonify({"message": "Emotion Detection API is running! Use /predict route."})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    try:
        results = emotion_model(text)[0]  # list of dicts with 'label' and 'score'
        top_label = max(results, key=lambda x: x['score'])
        probs = {r['label']: r['score'] for r in results}

        return jsonify({
            "label": top_label['label'],
            "probs": probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
