import joblib
import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import Dict, Union, Optional, List
from pathlib import Path

# Import the AdvancedFeatureEngineering class from the improved_model module
from improved_model import AdvancedFeatureEngineering

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coffee_predictions.log')
    ]
)
logger = logging.getLogger(__name__)

class CoffeeFlavorPredictor:
    """A class to handle coffee flavor predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize the predictor with a trained model."""
        self.model = None
        self.scaler = None
        self.feature_engineer = None
        self.selector = None
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []
        self.feature_engineering = None
        self.expected_columns = [
            'Brewing_Method', 'Bean_Type', 'Roast_Level', 'Grind_Size',
            'Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio',
            'Acidity_Pref', 'Bitterness_Pref'
        ]
        
        # Initialize feature engineering
        self.feature_engineering = AdvancedFeatureEngineering()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model with metadata
            model_data = joblib.load(model_path)
            
            # If it's the new format with metadata
            if isinstance(model_data, dict) and 'model' in model_data:
                # Store the model
                self.model = model_data['model']
                
                # Store preprocessing components with validation
                self.scaler = model_data.get('scaler')
                if self.scaler is None:
                    raise ValueError("Scaler not found in saved model data")
                
                self.feature_engineer = model_data.get('feature_engineer')
                self.selector = model_data.get('selector')
                self.numeric_features = model_data.get('numeric_features', [])
                self.categorical_features = model_data.get('categorical_features', [])
                self.feature_names = model_data.get('feature_names', [])
                
                # Validate that all required components are present
                if not self.feature_names:
                    raise ValueError("Feature names not found in saved model data")
                
                # If feature engineer is not in model_data, create a new one
                if self.feature_engineer is None:
                    self.feature_engineer = AdvancedFeatureEngineering()
                
                logger.info(f"Model loaded (new format). Type: {type(self.model).__name__}")
                logger.info(f"Scaler loaded: {self.scaler is not None}")
                logger.info(f"Feature engineer loaded: {self.feature_engineer is not None}")
                logger.info(f"Feature names count: {len(self.feature_names)}")
            else:
                # For backward compatibility with old format
                self.model = model_data
                if hasattr(self.model, 'feature_engineering'):
                    self.feature_engineering = self.model.feature_engineering
            
            logger.info(f"Successfully loaded model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
    
    def validate_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Validate and convert input data to DataFrame.
        
        Args:
            input_data: Dictionary containing the input features
            
        Returns:
            pd.DataFrame: Processed and validated DataFrame
            
        Raises:
            ValueError: If input validation fails
        """
        # Check for missing required fields
        missing_fields = [field for field in self.expected_columns if field not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Type conversion and validation
        numeric_fields = {
            'Water_Temp_C': (85, 100),  # Typical coffee brewing temperature range
            'Brew_Time_sec': (30, 600),  # 30 seconds to 10 minutes
            'Coffee_Water_Ratio': (0.04, 0.08),  # 4-8g per 100ml
            'Acidity_Pref': (1, 10),  # 1-10 scale
            'Bitterness_Pref': (1, 10)  # 1-10 scale
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            try:
                df[field] = pd.to_numeric(df[field])
                # Check if value is within expected range
                if not (min_val <= df[field].iloc[0] <= max_val):
                    logger.warning(f"{field} value {df[field].iloc[0]} is outside the expected range ({min_val}-{max_val})")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {field}: {input_data[field]}. Must be numeric.")
        
        # Validate categorical fields
        categorical_fields = {
            'Brewing_Method': ['Pour-over', 'French Press', 'Espresso', 'Cold Brew'],
            'Bean_Type': ['Arabica', 'Robusta', 'Blend'],
            'Roast_Level': ['Light', 'Medium', 'Dark'],
            'Grind_Size': ['Fine', 'Medium', 'Coarse']
        }
        
        for field, allowed_values in categorical_fields.items():
            if field in df.columns and df[field].iloc[0] not in allowed_values:
                logger.warning(f"{field} value '{df[field].iloc[0]}' is not in the list of standard values: {', '.join(allowed_values)}")
        
        return df
    
    def predict(self, input_data: Dict) -> Dict:
        """
        Make a prediction for the given input data.
        
        Args:
            input_data: Dictionary containing the input features
            
        Returns:
            Dict: {
                'prediction': float,  # Predicted score (1-10)
                'confidence': float,  # Confidence score (0-1)
                'interpretation': str,  # Human-readable interpretation
                'feature_importance': dict,  # Top important features
                'suggestions': List[str]  # List of suggestions for improvement
            }
            
        Raises:
            ValueError: If prediction fails
        """
        if not self.model:
            raise ValueError("No model loaded. Please load a model first.")
        
        if not self.scaler:
            raise ValueError("No scaler loaded. Please load a model with preprocessing components.")
        
        try:
            # Validate and convert input
            input_df = self.validate_input(input_data)
            
            # Apply the same preprocessing pipeline as training
            from improved_model import preprocess_data
            
            # Step 1: Preprocess the data (scaling and one-hot encoding)
            input_processed, _, _ = preprocess_data(
                input_df, 
                scaler=self.scaler, 
                fit_scaler=False, 
                feature_columns=self.feature_names
            )
            
            # Step 2: Use the preprocessed data directly (it already includes engineered features)
            input_final = input_processed
            
            # Convert to numpy array if it's a DataFrame (XGBoost expects numpy arrays)
            if hasattr(input_final, 'values'):
                input_final = input_final.values
            
            # Make prediction
            prediction = self.model.predict(input_final)[0]
            
            # Clip to reasonable range based on training data (1-10 scale)
            prediction = np.clip(prediction, 1, 10)
            
            # Calculate confidence based on prediction range and feature importance
            base_confidence = min(0.95, (prediction - 1) / 9 * 0.9 + 0.05)
            
            # Get feature importance if available (from XGBoost component)
            feature_importance = {}
            try:
                if hasattr(self.model, 'estimators_'):
                    xgb_model = None
                    for name, estimator in self.model.estimators_:
                        if name == 'xgb':
                            xgb_model = estimator
                            break
                    
                    if xgb_model is not None and hasattr(xgb_model, 'feature_importances_'):
                        importances = xgb_model.feature_importances_
                        if self.feature_names:
                            feature_importance_dict = dict(zip(self.feature_names, importances))
                            # Get top 5 features
                            sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                            feature_importance = dict(sorted_importance[:5])
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {str(e)}")
            
            # Generate suggestions for improvement
            suggestions = self._generate_suggestions(input_data, prediction, feature_importance)
            
            return {
                'prediction': round(prediction, 2),
                'confidence': round(base_confidence, 2),
                'interpretation': self.interpret_prediction(prediction),
                'feature_importance': feature_importance,
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def _generate_suggestions(self, input_data: Dict, prediction: float, feature_importance: Dict) -> List[str]:
        """Generate suggestions for improving the coffee flavor."""
        suggestions = []
        
        # Check water temperature
        temp = input_data.get('Water_Temp_C')
        if temp is not None:
            if temp < 90:
                suggestions.append("Consider increasing water temperature (90-96°C is optimal for most brews).")
            elif temp > 96:
                suggestions.append("Consider slightly reducing water temperature to avoid over-extraction.")
        
        # Check brew time
        brew_time = input_data.get('Brew_Time_sec')
        if brew_time is not None:
            if brew_time < 120 and input_data.get('Brewing_Method') != 'Espresso':
                suggestions.append("Try increasing brew time for better extraction (2-4 minutes is typical).")
            elif brew_time > 300 and input_data.get('Brewing_Method') not in ['Cold Brew', 'French Press']:
                suggestions.append("Brew time seems long - you might be over-extracting.")
        
        # Check coffee to water ratio
        ratio = input_data.get('Coffee_Water_Ratio')
        if ratio is not None:
            if ratio < 0.05:
                suggestions.append("Try using more coffee for better flavor (recommended ratio: 5-7g per 100ml).")
            elif ratio > 0.08:
                suggestions.append("Your coffee might be too strong. Try reducing the amount of coffee slightly.")
        
        # If prediction is low, provide general suggestions
        if prediction < 5:
            suggestions.extend([
                "Try using freshly roasted beans for better flavor.",
                "Ensure your coffee is ground just before brewing for maximum freshness.",
                "Make sure your water is filtered and free from strong odors."
            ])
        
        return suggestions
    
    @staticmethod
    def interpret_prediction(score: float) -> str:
        """
        Provide a human-readable interpretation of the prediction.
        
        Args:
            score: Predicted flavor score (1-10)
            
        Returns:
            str: Human-readable interpretation
        """
        if score >= 8.0:
            return "Exceptional coffee! This is among the best brews with well-balanced flavors and excellent extraction."
        elif score >= 7.0:
            return "Excellent coffee! You've got a great combination of flavors and extraction."
        elif score >= 6.0:
            return "Very good coffee. Most people would enjoy this well-balanced cup."
        elif score >= 5.0:
            return "Good coffee. It's drinkable but could use some adjustments for better flavor."
        elif score >= 4.0:
            return "Average coffee. Consider adjusting your brewing parameters for better results."
        else:
            return "Below average. Try adjusting your brewing parameters and check coffee freshness."

def get_example_input() -> Dict:
    """
    Return an example input for prediction with realistic values.
    
    Returns:
        Dict: Example input with realistic coffee brewing parameters
    """
    return {
        'Brewing_Method': 'Pour-over',  # One of: 'Pour-over', 'French Press', 'Espresso', 'Cold Brew'
        'Bean_Type': 'Arabica',         # One of: 'Arabica', 'Robusta', 'Blend'
        'Roast_Level': 'Medium',        # One of: 'Light', 'Medium', 'Dark'
        'Grind_Size': 'Medium',         # One of: 'Fine', 'Medium', 'Coarse'
        'Water_Temp_C': 92.0,           # Typically 85-100°C
        'Brew_Time_sec': 210,           # Typically 30-300 seconds (5 min)
        'Coffee_Water_Ratio': 0.065,    # Typically 0.04-0.08 (4-8g per 100ml)
        'Acidity_Pref': 6.0,            # 1-10 scale
        'Bitterness_Pref': 4.0          # 1-10 scale
    }

def print_prediction_result(result: Dict):
    """Print the prediction results in a user-friendly format."""
    print("\n" + "="*50)
    print("COFFEE FLAVOR PREDICTION RESULTS")
    print("="*50)
    
    # Print score and confidence
    print(f"\n{'Score:':<20} {result['prediction']:.1f}/10")
    print(f"{'Confidence:':<20} {result['confidence']*100:.0f}%")
    
    # Print interpretation
    print("\n" + "-"*50)
    print("FLAVOR ASSESSMENT")
    print("-"*50)
    print(result['interpretation'])
    
    # Print suggestions if available
    if 'suggestions' in result and result['suggestions']:
        print("\n" + "-"*50)
        print("SUGGESTIONS FOR IMPROVEMENT")
        print("-"*50)
        for i, suggestion in enumerate(result['suggestions'], 1):
            print(f"{i}. {suggestion}")
    
    # Print feature importance if available
    if 'feature_importance' in result and result['feature_importance']:
        print("\n" + "-"*50)
        print("MOST IMPORTANT FACTORS")
        print("-"*50)
        for feature, importance in result['feature_importance'].items():
            print(f"- {feature}: {importance:.2f}")
    
    print("\n" + "="*50 + "\n")

def main():
    """
    Run an example prediction.
    
    Returns:
        dict: Prediction results or None if an error occurred
    """
    try:
        # Initialize predictor
        predictor = CoffeeFlavorPredictor()
        
        # Find the latest model or train a new one
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_files = sorted(
            [f for f in os.listdir(model_dir) if f.endswith('.pkl')],
            key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
            reverse=True
        )
        
        if not model_files:
            logger.info("No trained models found. Training a new model...")
            from improved_model import main as train_model
            train_model()
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            if not model_files:
                raise RuntimeError("Failed to train a new model.")
        
        # Load the most recent model
        latest_model = os.path.join(model_dir, model_files[0])
        logger.info(f"Loading model: {latest_model}")
        predictor.load_model(latest_model)
        
        # Make a prediction with example data
        example_input = get_example_input()
        
        print("\nUsing example input parameters:")
        for key, value in example_input.items():
            print(f"- {key}: {value}")
        
        result = predictor.predict(example_input)
        
        # Print results
        print_prediction_result(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}\n")
        print("Please check the logs for more details.")
        return None

if __name__ == "__main__":
    main()
