import os
import joblib
import warnings
from flask import Flask, request, render_template, jsonify
# import joblib pandas etc.

warnings.filterwarnings('ignore')

app = Flask(__name__)
PORT = 5000

# Try load models on load
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
try:
    log_model = joblib.load(os.path.join(MODELS_DIR, 'Logistic_Regression.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'Random_Forest.pkl'))
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'SVM.pkl'))
    nb_model = joblib.load(os.path.join(MODELS_DIR, 'Naive_Bayes.pkl'))
    dt_model = joblib.load(os.path.join(MODELS_DIR, 'Decision_Tree.pkl'))
    knn_model = joblib.load(os.path.join(MODELS_DIR, 'KNN.pkl'))
    models = {
        'Logistic Regression': log_model,
        'Random Forest': rf_model,
        'SVM': svm_model,
        'Naive Bayes': nb_model,
        'Decision Tree': dt_model,
        'KNN': knn_model
    }
except Exception as e:
    print(f"Warning: Models not loaded. Please run train.py first. Error: {e}")
    models = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models:
         return jsonify({'error': 'Models are not trained. Please run training script.'}), 500

    job_desc = request.form.get('job_description', '')
    selected_model_name = request.form.get('model', 'Logistic Regression')

    if not job_desc.strip():
        return jsonify({'error': 'Job description is required.'}), 400

    selected_model = models.get(selected_model_name)
    if not selected_model:
        return jsonify({'error': 'Selected model is invalid.'}), 400
        
        
    try:
        prediction = selected_model.predict([job_desc])
        is_fake = bool(prediction[0])
        return jsonify({
            'is_fake': is_fake,
            'model': selected_model_name,
            'prediction_text': 'FAKE' if is_fake else 'REAL'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=PORT)
