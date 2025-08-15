from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sys
import logging
import joblib
import importlib.util
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import the AdvancedFeatureEngineering class to make it available during unpickling
from improved_model import AdvancedFeatureEngineering

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-for-coffee-app')

# Initialize the predictor and load the latest model
def load_latest_model():
    try:
        # First, make sure the model directory exists
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        # Find the latest model
        model_files = sorted(
            [f for f in os.listdir(model_dir) if f.endswith('.pkl')],
            key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
            reverse=True
        )
        
        if not model_files:
            logger.warning("No trained models found. Please train a model first.")
            return None
        
        # Import the predictor class here to ensure proper module context
        from improved_predict import CoffeeFlavorPredictor
        
        # Initialize the predictor
        predictor = CoffeeFlavorPredictor()
        
        # Get the latest model path
        latest_model = os.path.join(model_dir, model_files[0])
        logger.info(f"Loading model: {latest_model}")
        
        try:
            # Try to load the model using joblib
            import joblib
            model_data = joblib.load(latest_model)
            
            # Check the structure of the loaded data
            if isinstance(model_data, dict):
                if 'model' in model_data:
                    # New format with model and class definition
                    predictor.model = model_data['model']
                    logger.info(f"Model loaded (new format). Type: {type(predictor.model).__name__}")
                elif 'class_def' in model_data:
                    # Alternative format with class definition and model
                    predictor.model = model_data['model']
                    logger.info(f"Model loaded (alternative format). Type: {type(predictor.model).__name__}")
                else:
                    # Unknown dictionary format
                    logger.error(f"Unexpected model data format. Keys: {list(model_data.keys())}")
                    return None
            else:
                # Direct model (old format)
                predictor.model = model_data
                logger.info(f"Model loaded (old format). Type: {type(predictor.model).__name__}")
            
            # Verify the model has the required methods
            if not hasattr(predictor.model, 'predict'):
                logger.error("Loaded model does not have a 'predict' method")
                return None
                
            logger.info("Model loaded and verified successfully!")
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return None
        
    except Exception as e:
        logger.error(f"Error initializing predictor: {str(e)}", exc_info=True)
        return None

# Initialize the predictor
predictor = load_latest_model()
if predictor is None:
    logger.error("Failed to load any model. Please train a model first.")
else:
    logger.info("Model loaded and ready for predictions!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'No model loaded. Please train a model first.'
        }), 500
        
    try:
        # Get form data
        form_data = {
            'Brewing_Method': request.form.get('brewing_method'),
            'Bean_Type': request.form.get('bean_type'),
            'Roast_Level': request.form.get('roast_level'),
            'Grind_Size': request.form.get('grind_size'),
            'Water_Temp_C': float(request.form.get('water_temp')),
            'Brew_Time_sec': int(request.form.get('brew_time')),
            'Coffee_Water_Ratio': float(request.form.get('coffee_water_ratio')),
            'Acidity_Pref': float(request.form.get('acidity_pref')),
            'Bitterness_Pref': float(request.form.get('bitterness_pref'))
        }
        
        # Make prediction
        result = predictor.predict(form_data)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'score': round(result['prediction'], 1),
                'confidence': round(result['confidence'] * 100, 1),
                'interpretation': result['interpretation']
            }
        }
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An error occurred while making the prediction.'
        }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8000)
