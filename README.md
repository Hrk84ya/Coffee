# ☕ Coffee Flavor Prediction Model

A machine learning model that predicts coffee flavor scores based on brewing parameters, bean characteristics, and personal taste preferences. This tool helps coffee enthusiasts and professionals optimize their brewing process for the perfect cup.

## 🚀 Features

- **Accurate Predictions**: Predicts coffee flavor scores (1-10) with high precision
- **Comprehensive Parameters**: Considers various factors including:
  - Brewing method (e.g., Pour-over, French Press, Espresso)
  - Bean type and origin
  - Roast level and grind size
  - Water temperature and brew time
  - Personal taste preferences (acidity, bitterness)
- **Advanced Feature Engineering**: Automatic generation of meaningful features for better predictions
- **Model Interpretability**: Understand what factors most influence flavor
- **Easy Integration**: Simple Python API and command-line interface
- **Production Ready**: Built with scalability and maintainability in mind

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/coffee-flavor-prediction.git
   cd coffee-flavor-prediction
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

### Training the Model

To train a new model with default settings:

```bash
python improved_model.py
```

**What happens during training:**
1. Data loading and preprocessing
2. Feature engineering and transformation
3. Model training with cross-validation
4. Performance evaluation
5. Saving the trained model to `models/` directory

### Making Predictions

#### Using Python API

```python
from improved_predict import CoffeeFlavorPredictor

# Initialize predictor (automatically loads the latest model)
predictor = CoffeeFlavorPredictor()

# Example input
example_input = {
    'Brewing_Method': 'Pour-over',      # Options: ['Pour-over', 'French Press', 'Aeropress', 'Espresso', 'Moka Pot']
    'Bean_Type': 'Arabica',             # Options: ['Arabica', 'Robusta', 'Liberica', 'Excelsa']
    'Roast_Level': 'Medium',            # Options: ['Light', 'Medium', 'Dark']
    'Grind_Size': 'Medium',             # Options: ['Extra Fine', 'Fine', 'Medium', 'Coarse']
    'Water_Temp_C': 92.0,               # Range: 85-100°C
    'Brew_Time_sec': 210,               # Range: 15-300 seconds
    'Coffee_Water_Ratio': 0.065,        # Coffee to water ratio (e.g., 0.065 for 1:15.4 ratio)
    'Acidity_Pref': 6.0,                # 1-10 scale (1=low acidity, 10=high acidity)
    'Bitterness_Pref': 4.0              # 1-10 scale (1=low bitterness, 10=high bitterness)
}

# Get prediction
result = predictor.predict(example_input)
print(f"\n🔮 Prediction Results:")
print(f"  - Predicted Score: {result['prediction']:.1f}/10")
print(f"  - Confidence: {result['confidence']*100:.0f}%")
print(f"  - Interpretation: {result['interpretation']}")
```

#### Using Command Line

```bash
python improved_predict.py
# Follow the interactive prompts to input your parameters
```

## 📊 Model Performance

The model's performance is evaluated using multiple metrics:

| Metric | Value | Description |
|--------|-------|-------------|
| RMSE   | ~0.45 | Lower is better (scale 1-10) |
| R²     | ~0.92 | 1.0 is perfect prediction |
| MAE    | ~0.35 | Mean Absolute Error |
| Cross-Validated R² | ~0.89 | Performance on unseen data |

## 🏗️ Project Structure

```
coffee-flavor-prediction/
├── data/                           # Data directory
│   └── synthetic_coffee_dataset.csv  # Training dataset
├── models/                         # Saved models
│   ├── coffee_flavor_model_*.pkl   # Trained model files
├── improved_model.py              # Model training and evaluation
├── improved_predict.py            # Prediction interface
├── test_predictions.py            # Test script for predictions
├── requirements.txt               # Project dependencies
├── coffee_model.log               # Training logs
├── coffee_predictions.log         # Prediction logs
└── README.md                      # This file
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: Open an issue with detailed steps to reproduce
2. **Suggest Enhancements**: Share your ideas for new features
3. **Contribute Code**:
   ```bash
   # 1. Fork the repository
   # 2. Create your feature branch
   git checkout -b feature/amazing-feature
   # 3. Commit your changes
   git commit -m 'Add some amazing feature'
   # 4. Push to the branch
   git push origin feature/amazing-feature
   # 5. Open a Pull Request
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for coffee lovers worldwide
- Inspired by the quest for the perfect cup of coffee

---

<div align="center">
  <p>Made with ☕ and ❤️</p>
  <p>Happy Brewing! 🚀</p>
</div>
