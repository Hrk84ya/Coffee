# ☕ Coffee Flavor Predictor

An interactive web application that predicts coffee flavor profiles using machine learning. This tool helps coffee enthusiasts and professionals optimize their brewing process for the perfect cup, with a beautiful and intuitive interface.

## 🌟 Features

- **Web Interface**: Modern, responsive design for easy access from any device
- **Flavor Prediction**: Get instant flavor score predictions (1-10) based on your brew
- **Interactive Dashboard**: Visual feedback on flavor profiles and brewing parameters
- **Comprehensive Parameters**:
  - Multiple brewing methods (Pour-over, French Press, Espresso, etc.)
  - Various bean types and roast levels
  - Precise control over grind size, water temperature, and brew time
  - Personal taste preferences (acidity, bitterness)
- **Machine Learning**: Powered by XGBoost for accurate predictions
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/coffee-flavor-predictor.git
   cd coffee-flavor-predictor
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open in browser**:
   Visit `http://localhost:8000` in your web browser

## 🎯 Features in Action

### Web Interface

1. **Input Your Brew Parameters**:
   - Select your brewing method, bean type, and roast level
   - Adjust grind size, water temperature, and brew time
   - Set your personal taste preferences

2. **Get Instant Predictions**:
   - View your predicted flavor score (1-10)
   - See flavor profile breakdown
   - Get brewing tips based on your inputs

### Programmatic Usage

You can also use the prediction model directly in your Python code:

```python
from improved_predict import CoffeeFlavorPredictor

# Initialize predictor
predictor = CoffeeFlavorPredictor()

# Make a prediction
result = predictor.predict({
    'Brewing_Method': 'Pour-over',
    'Bean_Type': 'Arabica',
    'Roast_Level': 'Medium',
    'Grind_Size': 'Medium',
    'Water_Temp_C': 92.0,
    'Brew_Time_sec': 210,
    'Coffee_Water_Ratio': 0.0625,
    'Acidity_Pref': 6.0,
    'Bitterness_Pref': 4.0
})

print(f"Predicted Score: {result['prediction']:.1f}/10")
```

## 🏆 Model Performance

The underlying machine learning model has been rigorously evaluated:

| Metric | Score | Description |
|--------|-------|-------------|
| RMSE   | 0.42  | Lower is better |
| R²     | 0.93  | 1.0 is perfect prediction |
| MAE    | 0.32  | Mean Absolute Error |
| Cross-Validated R² | 0.90 | Performance on unseen data |

## 🏗️ Project Structure

```
coffee-flavor-predictor/
├── static/                        # Static files (CSS, JS, images)
│   ├── css/                      # Stylesheets
│   ├── js/                       # JavaScript files
│   └── images/                   # Image assets
├── templates/                    # HTML templates
│   ├── base.html                # Base template
│   └── index.html               # Main application page
├── models/                       # Saved ML models
│   └── coffee_flavor_model_*.pkl
├── app.py                       # Flask application
├── improved_model.py            # Model training
├── improved_predict.py          # Prediction logic
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🤝 How to Contribute

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug? Let us know!
2. **Feature Requests**: Have an idea? We'd love to hear it!
3. **Code Contributions**:
   ```bash
   # 1. Fork & clone the repo
   git clone https://github.com/yourusername/coffee-flavor-predictor.git
   cd coffee-flavor-predictor
   
   # 2. Create a feature branch
   git checkout -b feature/amazing-feature
   
   # 3. Make your changes
   # ...
   
   # 4. Commit and push
   git commit -m 'Add amazing feature'
   git push origin feature/amazing-feature
   
   # 5. Open a Pull Request
   ```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for coffee enthusiasts and professionals
- Special thanks to the open-source community for the amazing tools and libraries

---

<div align="center">
  <p>☕ Brewed with passion, powered by AI</p>
  <p>Happy Brewing! 🚀</p>
</div>
