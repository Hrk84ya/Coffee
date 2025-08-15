import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coffee_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """Enhanced feature engineering for coffee flavor prediction."""
    
    def __init__(self):
        self.feature_names = None
        self.numeric_features = ['Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio', 
                               'Acidity_Pref', 'Bitterness_Pref']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            # This is the case when using the pipeline
            X_df = pd.DataFrame(X, columns=self.numeric_features)
        else:
            # This is the case during prediction
            X_df = X.copy()
        
        # Create new features
        features = {}
        
        # Basic transformations
        if 'Coffee_Water_Ratio' in X_df.columns and 'Brew_Time_sec' in X_df.columns:
            features['extraction_yield'] = X_df['Coffee_Water_Ratio'] * X_df['Brew_Time_sec']
        if 'Water_Temp_C' in X_df.columns and 'Coffee_Water_Ratio' in X_df.columns:
            features['brew_strength'] = X_df['Water_Temp_C'] * X_df['Coffee_Water_Ratio']
        
        # Interaction terms
        if 'Water_Temp_C' in X_df.columns and 'Brew_Time_sec' in X_df.columns:
            features['temp_time_ratio'] = X_df['Water_Temp_C'] / (X_df['Brew_Time_sec'] + 1e-6)
        if 'Acidity_Pref' in X_df.columns and 'Bitterness_Pref' in X_df.columns:
            features['flavor_balance'] = (X_df['Acidity_Pref'] + 1) / (X_df['Bitterness_Pref'] + 1)
        
        # Polynomial features
        for col in ['Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio']:
            if col in X_df.columns:
                features[f'{col}_squared'] = X_df[col] ** 2
                features[f'{col}_sqrt'] = np.sqrt(np.abs(X_df[col]))
        
        # Log transforms
        if 'Brew_Time_sec' in X_df.columns:
            features['log_brew_time'] = np.log1p(X_df['Brew_Time_sec'])
        
        # Convert features to DataFrame
        new_features = pd.DataFrame(features)
        
        # If input was a numpy array, convert back to array
        if isinstance(X, np.ndarray):
            return np.hstack([X, new_features])
        
        # Otherwise, concatenate with original DataFrame
        return pd.concat([X_df, new_features], axis=1)

def load_and_preprocess_data(filepath='synthetic_coffee_dataset.csv'):
    """Load and preprocess the coffee dataset."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Basic validation
    required_columns = [
        'Brewing_Method', 'Bean_Type', 'Roast_Level', 'Grind_Size',
        'Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio',
        'Acidity_Pref', 'Bitterness_Pref', 'Flavor_Score'
    ]
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in the dataset")
    
    # Separate features and target
    X = df.drop('Flavor_Score', axis=1)
    y = df['Flavor_Score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df[['Bean_Type', 'Roast_Level']]
    )
    
    logger.info(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def create_preprocessor():
    """Create preprocessing pipeline."""
    numeric_features = ['Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio', 
                      'Acidity_Pref', 'Bitterness_Pref']
    categorical_features = ['Brewing_Method', 'Bean_Type', 'Roast_Level', 'Grind_Size']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_model(X_train, y_train):
    """Train and tune the model."""
    logger.info("Starting model training...")
    
    # Get numeric and categorical features
    numeric_features = ['Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio', 
                       'Acidity_Pref', 'Bitterness_Pref']
    categorical_features = ['Brewing_Method', 'Bean_Type', 'Roast_Level', 'Grind_Size']
    
    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # This will pass through any extra columns
    )
    
    # Initialize feature engineering with column names
    feature_engineering = AdvancedFeatureEngineering()
    
    # Define model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Apply feature engineering to training data before fitting
    X_train_engineered = feature_engineering.transform(X_train)
    
    # Train the model
    model.fit(X_train_engineered, y_train)
    
    # Add feature engineering to the model for later use in prediction
    model.feature_engineering = feature_engineering
    
    logger.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log metrics."""
    logger.info("Evaluating model...")
    
    # Apply feature engineering to test data
    if hasattr(model, 'feature_engineering'):
        X_test_engineered = model.feature_engineering.transform(X_test)
    else:
        X_test_engineered = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_engineered)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    logger.info(f"Test MAPE: {mape:.2f}%")
    
    # Feature importance
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        try:
            # Get feature names after preprocessing
            if hasattr(model, 'feature_engineering') and hasattr(model.feature_engineering, 'feature_names'):
                feature_names = model.feature_engineering.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(X_test_engineered.shape[1])]
            
            # Get feature importances
            importances = model.named_steps['regressor'].feature_importances_
            
            # Sort feature importances
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            logger.info("\nTop 10 important features:")
            for i in indices:
                if i < len(feature_names):
                    logger.info(f"{feature_names[i]}: {importances[i]:.4f}")
                else:
                    logger.info(f"feature_{i}: {importances[i]:.4f}")
                    
        except Exception as e:
            logger.warning(f"Could not log feature importances: {str(e)}")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'predictions': y_pred,
        'actuals': y_test
    }

def save_model(model, model_dir='models'):
    """Save the trained model and preprocessing pipeline."""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'coffee_flavor_model_{timestamp}.pkl')
    
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path

def main():
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = save_model(model)
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Final Test RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    model, metrics = main()
