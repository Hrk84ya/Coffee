import os
import sys
import joblib
from improved_predict import CoffeeFlavorPredictor, get_example_input
from improved_model import AdvancedFeatureEngineering

def run_tests():
    """Run a series of test predictions to verify the model works correctly."""
    print("Starting prediction tests...\n")
    
    # Initialize predictor and load the latest model
    predictor = CoffeeFlavorPredictor()
    model_dir = 'models'
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')], reverse=True)
    
    if not model_files:
        print("Error: No model found for testing")
        return False
    
    print(f"Loading model: {model_files[0]}")
    predictor.load_model(os.path.join(model_dir, model_files[0]))
    
    # Test 1: Default example input
    print("\n=== Test 1: Default Example Input ===")
    input_data = get_example_input()
    result = predictor.predict(input_data)
    print_result(input_data, result)
    
    # Test 2: Lower temperature
    print("\n=== Test 2: Lower Temperature ===")
    input_data = get_example_input()
    input_data['Water_Temp_C'] = 88.0
    result = predictor.predict(input_data)
    print_result(input_data, result)
    
    # Test 3: French Press with coarse grind
    print("\n=== Test 3: French Press ===")
    input_data = get_example_input()
    input_data.update({
        'Brewing_Method': 'French Press',
        'Grind_Size': 'Coarse',
        'Brew_Time_sec': 240
    })
    result = predictor.predict(input_data)
    print_result(input_data, result)
    
    # Test 4: Dark roast with higher bitterness preference
    print("\n=== Test 4: Dark Roast ===")
    input_data = get_example_input()
    input_data.update({
        'Roast_Level': 'Dark',
        'Bitterness_Pref': 8.0,
        'Acidity_Pref': 3.0
    })
    result = predictor.predict(input_data)
    print_result(input_data, result)
    
    print("\nAll tests completed successfully!")
    return True

def print_result(input_data, result):
    """Print prediction results in a readable format."""
    print(f"\nInput Parameters:")
    for key, value in input_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction Results:")
    print(f"  Score: {result['prediction']:.1f}/10")
    print(f"  Confidence: {result['confidence']*100:.0f}%")
    print(f"  Interpretation: {result['interpretation']}")
    
    if 'suggestions' in result and result['suggestions']:
        print("\n  Suggestions:")
        for i, suggestion in enumerate(result['suggestions'], 1):
            print(f"  {i}. {suggestion}")

if __name__ == "__main__":
    run_tests()
