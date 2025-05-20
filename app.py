import os
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Important: We need to import InferenceModel from your package.
# For this, Python needs to know about the transformer_text_classification package.
# Option 1: Add the path to sys.path (less preferred for project structure)
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "transformer_text_classification"))
# from src.inference_model import InferenceModel

# Option 2: Ensure transformer_text_classification is an installable package
# or you are running Flask from an environment where it can find this package.
# Assuming you will run Flask from the transformer_application root directory
# and Python can find transformer_text_classification.src
from transformer_text_classification.src.inference_model import InferenceModel


app = Flask(__name__)

# --- Configuration ---
MODEL_DIR = "saved_model_and_tokenizer" # Directory containing the saved model and tokenizer
# Absolute path to the model directory
SAVED_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_DIR)

MAX_LENGTH = 128 # Or get from your config if needed, e.g., CONFIG["max_length"]

# --- Global variables for model and tokenizer ---
predictor = None
device = None

def load_model_resources():
    """Loads the model, tokenizer, and creates the predictor instance."""
    global predictor, device

    if not os.path.exists(SAVED_MODEL_PATH) or not os.listdir(SAVED_MODEL_PATH):
        print(f"Error: Saved model directory '{SAVED_MODEL_PATH}' not found or is empty.")
        print("Please ensure the model is trained and saved correctly.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model and tokenizer from: {SAVED_MODEL_PATH}")
    print(f"Using device: {device}")

    try:
        loaded_model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)
        loaded_tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH)
        loaded_model.to(device)

        predictor = InferenceModel(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device=device,
            max_length=MAX_LENGTH
        )
        print("Model, tokenizer, and predictor loaded successfully.")
    except Exception as e:
        print(f"Error loading model resources: {e}")
        predictor = None

# Call load_model_resources() when the module is imported by Gunicorn
# This should be after app = Flask(__name__) and before any routes if predictor is used by them directly,
# or simply at the module level like this.
load_model_resources() # <--- THAY ĐỔI QUAN TRỌNG: Gọi hàm này ở đây

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """Handles sentiment prediction requests."""
    if predictor is None:
        # This message will now be more relevant if loading failed
        return jsonify({"error": "Model not loaded or failed to load. Please check server logs."}), 500

    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request."}), 400
        
        text_to_predict = data['text']
        if not text_to_predict.strip():
            return jsonify({"error": "'text' field cannot be empty."}), 400

        prediction_result = predictor.predict([text_to_predict])
        
        if prediction_result:
            return jsonify(prediction_result[0]) 
        else:
            return jsonify({"error": "Prediction failed."}), 500
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Không cần gọi load_model_resources() ở đây nữa vì nó đã được gọi ở trên
    app.run(host='0.0.0.0', port=5000, debug=True)