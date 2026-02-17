# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import traceback
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080", "http://127.0.0.1:8080", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============ LOAD YOUR TRAINED MODEL ============
model = None
vectorizer = None

try:
    # Load your trained model
    model_path = os.path.join("models", "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ YOUR TRAINED MODEL LOADED: {type(model).__name__}")
        
        # Check if model is fitted
        if hasattr(model, 'classes_'):
            print(f"   ✅ Model is FITTED with classes: {model.classes_}")
            print(f"   📊 Number of features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'}")
            
            # Try to load vectorizer if it exists (for text models)
            vec_path = os.path.join("models", "vectorizer.pkl")
            if os.path.exists(vec_path):
                with open(vec_path, "rb") as f:
                    vectorizer = pickle.load(f)
                print(f"   ✅ Vectorizer loaded: {type(vectorizer).__name__}")
            else:
                print("   ⚠️ No vectorizer found - will use feature extraction")
        else:
            print("   ❌ Model is NOT fitted! Will retrain...")
            # Retrain a simple model
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Training data
            X_train = [
                "URGENT! Pay registration fee of ₹2000 to get job",
                "Work from home, earn ₹50000/month, send money for processing",
                "Congratulations! You won lottery, transfer fee to claim",
                "Job offer: Software Engineer at Google, interview on campus",
                "Thank you for applying, please find attached offer letter",
                "Interview scheduled for next week at our Bangalore office",
                "Send ₹5000 as security deposit to confirm your selection",
                "WhatsApp only: +91 98765 43210 for immediate joining",
                "We are pleased to offer you the position of Developer",
                "URGENT: Pay now or lose this opportunity forever"
            ]
            y_train = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]  # 1 = scam, 0 = genuine
            
            vectorizer = CountVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            
            model = MultinomialNB()
            model.fit(X_train_vec, y_train)
            print("   ✅ New model trained successfully!")
            
            # Save the new model
            with open("models/retrained_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("models/retrained_vectorizer.pkl", "wb") as f:
                pickle.dump(vectorizer, f)
            print("   💾 New model saved as 'retrained_model.pkl'")
    else:
        print("❌ No model.pkl found - using fallback detection")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()
    model = None
    vectorizer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    text = ""
    try:
        if filepath.lower().endswith('.pdf'):
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
    except Exception as e:
        print(f"Text extraction error: {e}")
    return text.strip()[:15000]

# ============ ACTUAL MODEL PREDICTION FUNCTION ============
def predict_with_trained_model(text):
    """
    THIS FUNCTION ACTUALLY USES YOUR TRAINED MODEL
    """
    global model, vectorizer
    
    # If no model, use fallback
    if model is None:
        print("⚠️ No model available - using fallback")
        return get_fallback_prediction(text)
    
    try:
        # Prepare features based on what your model expects
        if vectorizer is not None:
            # If you have a vectorizer, use it (for text models)
            features = vectorizer.transform([text])
            print(f"✅ Using vectorizer: features shape {features.shape}")
        else:
            # If your model expects numerical features
            # YOU NEED TO CUSTOMIZE THIS BASED ON HOW YOU TRAINED YOUR MODEL
            features = extract_numerical_features(text)
            print(f"✅ Using numerical features: shape {features.shape}")
        
        # Make prediction with YOUR trained model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]
            
            print(f"📊 Model probabilities: {probabilities}")
            print(f"🎯 Model prediction: {prediction}")
            
            # Map to trust score (assuming class 1 = scam)
            if len(probabilities) == 2:
                prob_scam = probabilities[1]  # Class 1 is scam
                prob_genuine = probabilities[0]
            else:
                prob_scam = probabilities[0]
            
            trust_score = int(round((1 - prob_scam) * 100))
            
        else:
            prediction = model.predict(features)[0]
            print(f"🎯 Model prediction: {prediction}")
            
            # Convert prediction to trust score
            if prediction in [1, 'scam', 'high_risk']:
                trust_score = 20
                prob_scam = 0.8
            elif prediction in [0, 'genuine', 'safe']:
                trust_score = 80
                prob_scam = 0.2
            else:
                trust_score = 50
                prob_scam = 0.5
        
        # Ensure trust score is within bounds
        trust_score = max(0, min(100, trust_score))
        
        # Generate warnings based on model output
        warnings = []
        if trust_score < 40:
            warnings.append(f"🚨 YOUR MODEL PREDICTS: HIGH RISK (confidence: {prob_scam:.0%})")
        elif trust_score < 70:
            warnings.append(f"⚠️ YOUR MODEL PREDICTS: MEDIUM RISK (confidence: {prob_scam:.0%})")
        else:
            warnings.append(f"✅ YOUR MODEL PREDICTS: LOW RISK (confidence: {1-prob_scam:.0%})")
        
        # Add specific warnings based on features
        if "pay" in text.lower() or "fee" in text.lower():
            warnings.append("💰 Payment keywords detected")
        if "whatsapp" in text.lower():
            warnings.append("📱 WhatsApp mention detected")
        
        positive_indicators = []
        if trust_score >= 70:
            positive_indicators.append(f"✅ Model confidence: High trust score")
        if "company" in text.lower() or "website" in text.lower():
            positive_indicators.append("✓ Company information present")
        
        return {
            "trust_score": trust_score,
            "prob_scam": prob_scam,
            "status": map_status_to_frontend(trust_score),
            "warnings": warnings[:3],  # Max 3 warnings
            "positive_indicators": positive_indicators[:2],
            "prediction": str(prediction),
            "model_used": "YOUR TRAINED MODEL",
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Model prediction error: {e}")
        traceback.print_exc()
        return get_fallback_prediction(text)

def extract_numerical_features(text):
    """
    Extract numerical features if your model expects them
    YOU MUST CUSTOMIZE THIS BASED ON YOUR TRAINING DATA
    """
    text_lower = text.lower()
    
    # Example features - REPLACE WITH YOUR ACTUAL FEATURES
    features = [
        len(text),  # text length
        len(text.split()),  # word count
        text_lower.count("urgent"),
        text_lower.count("pay"),
        text_lower.count("fee"),
        text_lower.count("bank"),
        text_lower.count("money"),
        1 if "whatsapp" in text_lower else 0,
        text_lower.count("register"),
        text_lower.count("transfer"),
        text_lower.count("job"),
        text_lower.count("offer"),
        text_lower.count("salary"),
        text_lower.count("interview"),
        1 if "@" in text else 0,  # has email
        1 if "http" in text_lower else 0,  # has URL
    ]
    
    return np.array(features).reshape(1, -1)

def get_fallback_prediction(text):
    """Fallback when model fails"""
    text_lower = text.lower()
    scam_keywords = ["urgent", "pay", "fee", "bank", "whatsapp", "money", "transfer", "register"]
    keyword_count = sum(1 for kw in scam_keywords if kw in text_lower)
    
    if keyword_count >= 3:
        trust_score = 30
        warnings = ["⚠️ FALLBACK: Multiple scam keywords detected"]
    elif keyword_count >= 1:
        trust_score = 60
        warnings = ["⚠️ FALLBACK: Some suspicious keywords detected"]
    else:
        trust_score = 85
        warnings = ["✅ FALLBACK: No obvious red flags"]
    
    return {
        "trust_score": trust_score,
        "prob_scam": (100 - trust_score) / 100,
        "status": map_status_to_frontend(trust_score),
        "warnings": warnings,
        "positive_indicators": ["Using fallback detection - model unavailable"],
        "model_used": "FALLBACK (keyword counting)",
        "success": True,
        "fallback": True
    }

def map_status_to_frontend(trust_score):
    if trust_score >= 70:
        return "genuine"
    elif trust_score >= 40:
        return "caution"
    else:
        return "scam"

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_fitted": hasattr(model, 'classes_') if model else False,
        "using_real_model": hasattr(model, 'classes_') if model else False,
        "detection_mode": "REAL TRAINED MODEL" if (model and hasattr(model, 'classes_')) else "FALLBACK"
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    saved_paths = []
    extracted_texts = []

    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                saved_paths.append(filepath)
                
                text = extract_text(filepath)
                if text:
                    extracted_texts.append(text)

        if not saved_paths:
            return jsonify({"error": "No valid files"}), 400

        combined_text = "\n\n".join(extracted_texts) if extracted_texts else "Document uploaded"
        
        # USE YOUR TRAINED MODEL HERE
        if model and hasattr(model, 'classes_'):
            prediction = predict_with_trained_model(combined_text)
            print("✅ Using REAL TRAINED MODEL for prediction")
        else:
            prediction = get_fallback_prediction(combined_text)
            print("⚠️ Using FALLBACK detection")

        result = {
            "trustScore": prediction['trust_score'],
            "status": prediction['status'],
            "probability_scam_percent": round(prediction['prob_scam'] * 100, 1),
            "warnings": prediction['warnings'],
            "positive_indicators": prediction['positive_indicators'],
            "recommendations": [
                "🔍 Verify company through official website",
                "💰 Never pay any money during hiring",
                "📧 Contact HR through official email",
                "🌐 Check for similar scam reports online"
            ],
            "extracted_text_length": len(combined_text),
            "files_processed": len(saved_paths),
            "model_used": prediction.get('model_used', 'unknown')
        }

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "Missing text"}), 400

    # USE YOUR TRAINED MODEL HERE
    if model and hasattr(model, 'classes_'):
        prediction = predict_with_trained_model(text)
        print("✅ Using REAL TRAINED MODEL for text analysis")
    else:
        prediction = get_fallback_prediction(text)
        print("⚠️ Using FALLBACK detection for text")

    result = {
        "trustScore": prediction['trust_score'],
        "status": prediction['status'],
        "probability_scam_percent": round(prediction['prob_scam'] * 100, 1),
        "warnings": prediction['warnings'],
        "positive_indicators": prediction['positive_indicators'],
        "recommendations": [
            "🔍 Verify company through official website",
            "💰 Never pay any money during hiring",
            "📧 Contact HR through official email",
            "🌐 Check for similar scam reports online"
        ],
        "model_used": prediction.get('model_used', 'unknown')
    }
    
    return jsonify(result), 200

@app.route('/debug/model-status', methods=['GET'])
def model_status():
    """Check if real model is being used"""
    return jsonify({
        "using_real_model": model is not None and hasattr(model, 'classes_'),
        "model_type": str(type(model)) if model else None,
        "is_fitted": hasattr(model, 'classes_') if model else False,
        "vectorizer_loaded": vectorizer is not None,
        "message": "✅ Using REAL trained model" if (model and hasattr(model, 'classes_')) else "⚠️ Using FALLBACK detection"
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 TRUSTHIRE BACKEND - MODEL STATUS")
    print("="*80)
    
    if model and hasattr(model, 'classes_'):
        print("✅✅✅ USING YOUR REAL TRAINED MODEL ✅✅✅")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Classes: {model.classes_}")
        if hasattr(model, 'n_features_in_'):
            print(f"   Features: {model.n_features_in_}")
    elif model:
        print("⚠️ Model loaded but NOT FITTED - using fallback")
    else:
        print("⚠️ No model loaded - using fallback detection")
    
    print("\n📡 Endpoints:")
    print("   GET  /debug/model-status  - Check if real model is used")
    print("   POST /api/analyze          - File upload")
    print("   POST /api/analyze-text     - Text input")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
# # app.py
# # TrustHire backend - with model loading support
# # Run: python app.py

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from werkzeug.utils import secure_filename
# import fitz  # PyMuPDF - pip install PyMuPDF

# from flask_cors import CORS
# CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})
# # ── Try to load model at startup ───────────────────────────────────────
# MODEL_PATH = os.path.join("models", "model.pkl")
# model = None

# try:
#     import pickle
#     if os.path.exists(MODEL_PATH):
#         with open(MODEL_PATH, 'rb') as f:
#             model = pickle.load(f)
#         print(f"✓ Model loaded successfully from {MODEL_PATH}")
#         print(f"   Model type: {type(model).__name__}")
#     else:
#         print(f"⚠ Model file not found: {MODEL_PATH}")
# except Exception as e:
#     print(f"✗ Failed to load model: {str(e)}")
#     model = None

# # ── Flask app ───────────────────────────────────────────────────────────
# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = "uploads"
# ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs("models", exist_ok=True)


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def extract_text_from_file(filepath):
#     """Very basic text extraction — mainly for PDF right now"""
#     text = ""
#     try:
#         if filepath.lower().endswith('.pdf'):
#             doc = fitz.open(filepath)
#             for page in doc:
#                 text += page.get_text("text") + "\n"
#             doc.close()
#         # You can add image OCR later with pytesseract
#         # else:  # png/jpg
#         #     text = pytesseract.image_to_string(Image.open(filepath))
#     except Exception as e:
#         print(f"Text extraction failed for {filepath}: {e}")
#     return text.strip()[:8000]  # limit length to avoid memory issues


# @app.route('/api/analyze', methods=['POST'])
# def analyze():
#     if 'files' not in request.files or not request.files.getlist('files'):
#         return jsonify({"error": "No files uploaded"}), 400

#     files = request.files.getlist('files')
#     job_title = request.form.get('job_title', '').strip()
#     company_name = request.form.get('company_name', '').strip()

#     saved_paths = []
#     extracted_texts = []

#     try:
#         # 1. Save files temporarily
#         for file in files:
#             if not file.filename or not allowed_file(file.filename):
#                 continue

#             filename = secure_filename(file.filename)
#             filepath = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(filepath)
#             saved_paths.append(filepath)

#             # 2. Extract text
#             text = extract_text_from_file(filepath)
#             if text:
#                 extracted_texts.append(text)

#         if not saved_paths:
#             return jsonify({"error": "No valid files processed"}), 400

#         full_text = "\n\n".join(extracted_texts)

#         # ── PREDICTION LOGIC ───────────────────────────────────────
#         if model is None:
#             # Fallback when model is not loaded
#             result = _get_mock_result(full_text, job_title, company_name)
#         else:
#             # ── REPLACE THIS PART WITH YOUR REAL FEATURE EXTRACTION ──
#             # Example: very naive approach — most models will NOT accept raw text
#             # You probably need TF-IDF, embeddings, regex features, etc.

#             # Placeholder: pretend we have features
#             # Real version example:
#             # from sklearn.feature_extraction.text import TfidfVectorizer
#             # vectorizer = ... (must be fitted & pickled together or loaded separately)
#             # features = vectorizer.transform([full_text])

#             features = [len(full_text), full_text.lower().count("urgent"), full_text.lower().count("pay")]  # dummy

#             try:
#                 # Most sklearn models expect 2D array
#                 if hasattr(model, 'predict_proba'):
#                     prob = model.predict_proba([features])[0]
#                     scam_prob = prob[1] if len(prob) > 1 else prob[0]  # assume class 1 = scam
#                 else:
#                     pred = model.predict([features])[0]
#                     scam_prob = float(pred) if isinstance(pred, (int, float)) else 0.5

#                 trust_score = max(0, min(100, int(100 - (scam_prob * 100))))
#             except Exception as e:
#                 print(f"Model prediction failed: {e}")
#                 trust_score = 50
#                 scam_prob = 0.5

#             result = {
#                 "trustScore": trust_score,
#                 "status": "Safe" if trust_score >= 80 else "Caution" if trust_score >= 50 else "High Risk",
#                 "probability_scam_percent": round(scam_prob * 100, 1),
#                 "warnings": ["Model-based detection active"] if trust_score < 70 else [],
#                 "positive_indicators": ["No obvious red flags in text length"] if trust_score >= 70 else [],
#                 "recommendations": [
#                     "Offer analyzed using ML model",
#                     "Verify sender email domain",
#                     "Check company registration"
#                 ],
#                 "extracted_text_length": len(full_text),
#                 "files_processed": len(saved_paths)
#             }

#         return jsonify(result), 200

#     except Exception as e:
#         return jsonify({"error": f"Server error: {str(e)}"}), 500

#     finally:
#         # Cleanup
#         for path in saved_paths:
#             try:
#                 if os.path.exists(path):
#                     os.remove(path)
#             except:
#                 pass


# def _get_mock_result(text, job_title, company):
#     """Fallback mock when model not available"""
#     score = 78
#     return {
#         "trustScore": score,
#         "status": "Probably Safe",
#         "probability_scam_percent": 22,
#         "warnings": ["Mock mode - real model not loaded"],
#         "positive_indicators": ["No payment words detected"],
#         "recommendations": ["Verify company directly", "Never pay upfront"],
#         "extracted_text_length_chars": len(text),
#         "job_title": job_title or None,
#         "company": company or None
#     }


# @app.route('/health', methods=['GET'])
# def health():
#     status = "Model loaded" if model is not None else "No model loaded (using mock)"
#     return jsonify({
#         "status": "ok",
#         "backend": "TrustHire",
#         "model_status": status,
#         "upload_folder": UPLOAD_FOLDER
#     })


# if __name__ == '__main__':
#     print("TrustHire backend starting...")
#     print(f"Model status: {'Loaded' if model else 'Not loaded - using mock'}")
#     print(" → http://localhost:5001/health")
#     print(" → POST http://localhost:5001/api/analyze")
#     app.run(debug=True, host='0.0.0.0', port=5001)


