# Coffee Flavor Prediction

A machine learning model to predict coffee flavor scores based on brewing parameters and preferences.

## Features

- Predicts coffee flavor scores (1-10) based on brewing parameters
- Handles various coffee types, roast levels, and brewing methods
- Includes feature engineering for better predictions
- Easy-to-use prediction interface
- Model evaluation and visualization tools

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd coffee-flavor-prediction
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train a new model:

```bash
python improved_model.py
```

This will:
1. Load and preprocess the data
2. Train the model with cross-validation
3. Save the trained model to the `models` directory
4. Print evaluation metrics

### Making Predictions

To make predictions using the trained model:

```python
from improved_predict import CoffeeFlavorPredictor

# Initialize the predictor with the latest model
predictor = CoffeeFlavorPredictor()

# Example input
example_input = {
    'Brewing_Method': 'Pour-over',
    'Bean_Type': 'Arabica',
    'Roast_Level': 'Medium',
    'Grind_Size': 'Medium',
    'Water_Temp_C': 92.0,
    'Brew_Time_sec': 210,
    'Coffee_Water_Ratio': 0.065,
    'Acidity_Pref': 6.0,
    'Bitterness_Pref': 4.0
}

# Make prediction
result = predictor.predict(example_input)
print(f"Predicted Score: {result['prediction']}/10")
print(f"Confidence: {result['confidence']*100:.0f}%")
print(f"Interpretation: {result['interpretation']}")
```

Or use the command line:

```bash
python improved_predict.py
```

## Project Structure

```
coffee-flavor-prediction/
├── data/
│   └── synthetic_coffee_dataset.csv  # Training data
├── models/                          # Trained models
├── improved_model.py               # Model training script
├── improved_predict.py             # Prediction interface
├── analyze_data.py                 # Data analysis utilities
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Model Performance

The model's performance is evaluated using:
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation results

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
