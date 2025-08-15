import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import the training function from improved_model
from improved_model import main as train_model

def main():
    print("Starting model training...")
    
    try:
        # Train a new model
        train_result = train_model()
        
        # The train_model function might return a tuple or just the model path
        if isinstance(train_result, tuple):
            model_path = train_result[0]  # Get the first item if it's a tuple
        else:
            model_path = train_result
            
        if model_path and os.path.exists(model_path):
            print(f"\nModel trained and saved to: {model_path}")
            print("You can now run the Flask application with the new model.")
            
            # Also print the parent directory for reference
            print(f"Model directory: {os.path.dirname(os.path.abspath(model_path))}")
            print(f"Files in model directory: {os.listdir(os.path.dirname(model_path))}")
            
        else:
            print("\nError: Model training failed or model was not saved correctly.")
            return 1
            
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
