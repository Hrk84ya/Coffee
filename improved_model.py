import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
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
    """Load and preprocess the dataset."""
    try:
        # Load the dataset
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Define categorical columns
        categorical_cols = ['Bean_Type', 'Roast_Level', 'Grind_Size', 'Brewing_Method']
        
        # Convert categorical columns to 'category' type
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Split into features and target
        X = df.drop('Flavor_Score', axis=1)
        y = df['Flavor_Score']
        
        # Split into train and test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df[categorical_cols]
        )
        
        logger.info(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Log value counts for categorical columns
        for col in categorical_cols:
            logger.info(f"\n{col} value counts (train):")
            logger.info(X_train[col].value_counts())
            
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

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

def preprocess_data(X):
    """Preprocess the input data by converting categorical columns to appropriate types."""
    X_processed = X.copy()
    categorical_features = ['Bean_Type', 'Roast_Level', 'Grind_Size', 'Brewing_Method']
    
    # Convert categorical columns to string type
    for col in categorical_features:
        if col in X_processed.columns:
            X_processed[col] = X_processed[col].astype(str)
    
    return X_processed

def preprocess_data(X, scaler=None, fit_scaler=False, feature_columns=None):
    """Preprocess the data with consistent handling of features."""
    # Define features
    numeric_features = ['Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio', 
                       'Acidity_Pref', 'Bitterness_Pref']
    categorical_features = ['Bean_Type', 'Roast_Level', 'Grind_Size', 'Brewing_Method']
    
    # Create a copy to avoid modifying the original data
    X_processed = X.copy()
    
    # Convert categorical columns to string type
    for col in categorical_features:
        if col in X_processed.columns:
            X_processed[col] = X_processed[col].astype(str)
    
    # Scale numeric features
    if fit_scaler:
        scaler = StandardScaler()
        X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
    else:
        X_processed[numeric_features] = scaler.transform(X_processed[numeric_features])
    
    # One-hot encode categorical features
    X_processed = pd.get_dummies(X_processed, columns=categorical_features)
    
    # If feature_columns is provided, ensure all columns are present
    if feature_columns is not None:
        missing_cols = set(feature_columns) - set(X_processed.columns)
        for col in missing_cols:
            X_processed[col] = 0
        X_processed = X_processed[feature_columns]
    
    return X_processed, scaler, X_processed.columns.tolist()

def train_model(X_train, y_train, use_cross_validation=True):
    """Train the XGBoost model with optional cross-validation."""
    try:
        # Define features
        numeric_features = ['Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio', 
                           'Acidity_Pref', 'Bitterness_Pref']
        categorical_features = ['Bean_Type', 'Roast_Level', 'Grind_Size', 'Brewing_Method']
        
        # First preprocess the training data and get the feature columns
        X_train_processed, scaler, feature_columns = preprocess_data(
            X_train, fit_scaler=True
        )
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1
        }
        
        if use_cross_validation:
            logger.info("Starting model training with cross-validation...")
            
            # Split original data for early stopping (before preprocessing)
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Preprocess the split training data
            X_train_split_processed, _, _ = preprocess_data(
                X_train_split, fit_scaler=True
            )
            
            # Preprocess validation data using the same scaler and feature columns
            X_val_processed, _, _ = preprocess_data(
                X_val, 
                scaler=scaler, 
                fit_scaler=False, 
                feature_columns=feature_columns
            )
            
            # Ensure all columns from training are present in validation
            missing_cols = set(feature_columns) - set(X_val_processed.columns)
            for col in missing_cols:
                X_val_processed[col] = 0
            
            # Reorder columns to match training data
            X_val_processed = X_val_processed[feature_columns]
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train_split_processed, label=y_train_split)
            dval = xgb.DMatrix(X_val_processed, label=y_val)
            
            # Train with early stopping
            evals_result = {}
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                early_stopping_rounds=20,
                evals_result=evals_result,
                verbose_eval=10
            )
            
            logger.info(f"Best iteration: {model.best_iteration}")
            logger.info(f"Best score: {model.best_score:.4f}")
            
        else:
            logger.info("Starting model training...")
            dtrain = xgb.DMatrix(X_train_processed, label=y_train)
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=100
            )
            logger.info("Model training completed")
        
        # Save the model and preprocessing objects
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(model_dir, f'coffee_flavor_model_{timestamp}.pkl')
        
        # Save the model and preprocessing objects
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'feature_names': X_train_processed.columns.tolist()
        }, model_file)
        
        logger.info(f"Model saved to {model_file}")
        
        # Log feature importance
        try:
            importance = model.get_score(importance_type='weight')
            if importance:
                logger.info("\nFeature Importance:")
                for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                    logger.info(f"{k}: {v:.4f}")
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise 

def evaluate_model(model, X_test, y_test, scaler, numeric_features, categorical_features, feature_names=None):
    """Evaluate the model and log metrics."""
    try:
        # Preprocess the test data using the same scaler and feature columns
        X_test_processed, _, _ = preprocess_data(
            X_test, 
            scaler=scaler, 
            fit_scaler=False, 
            feature_columns=feature_names
        )
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X_test_processed)
        
        # Make predictions
        y_pred = model.predict(dtest)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Mean Absolute Percentage Error
        
        logger.info(f"\nModel Evaluation Metrics:")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"MAPE: {mape:.2f}%")
        
        # Log feature importances
        try:
            importances = model.get_score(importance_type='weight')
            if importances:
                logger.info("\nTop 10 important features:")
                for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
                    logger.info(f"{feature}: {importance:.4f}")
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            
        return {
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred,
            'actuals': y_test.values
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise 

def save_model(model, model_dir='models'):
    """Save the trained model and preprocessing pipeline."""
    import sys
    import inspect
    
    # Get the source code of the AdvancedFeatureEngineering class
    from improved_model import AdvancedFeatureEngineering
    
    # Create a dictionary with the class definition
    model_metadata = {
        'class_def': {
            'module': AdvancedFeatureEngineering.__module__,
            'name': AdvancedFeatureEngineering.__name__,
            'source': inspect.getsource(AdvancedFeatureEngineering)
        },
        'model': model
    }
    
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'coffee_flavor_model_{timestamp}.pkl')
    
    # Save the model with metadata
    joblib.dump(model_metadata, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path

def main():
    """Main function to train and evaluate the model."""
    try:
        logger.info("Loading data...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        
        # Get feature names before preprocessing
        numeric_features = ['Water_Temp_C', 'Brew_Time_sec', 'Coffee_Water_Ratio', 
                          'Acidity_Pref', 'Bitterness_Pref']
        categorical_features = ['Bean_Type', 'Roast_Level', 'Grind_Size', 'Brewing_Method']
        
        logger.info("Training model...")
        model = train_model(X_train, y_train, use_cross_validation=True)
        
        # Load the saved model to get preprocessing objects
        model_dir = 'models'
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')], reverse=True)
        if model_files:
            latest_model = joblib.load(os.path.join(model_dir, model_files[0]))
            model = latest_model['model']
            scaler = latest_model['scaler']
            feature_names = latest_model['feature_names']
            
            logger.info("Evaluating model...")
            metrics = evaluate_model(
                model, X_test, y_test, 
                scaler=scaler,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                feature_names=feature_names
            )
            
            result = {
                'model': model,
                'metrics': {
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'mape': metrics['mape']
                },
                'dataset_info': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': feature_names
                }
            }
            
            logger.info("\nModel training and evaluation completed successfully!")
            return result
            
        else:
            logger.error("No trained model found for evaluation")
            return None
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    result = main()
    if result:
        print(f"Model trained successfully with R² score: {result['metrics']['r2']:.4f}")
    else:
        print("Model training failed")
