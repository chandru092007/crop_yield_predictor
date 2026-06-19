# Crop Yield Prediction (India)

A small Streamlit app and model for predicting crop yield (quintal per hectare) using agricultural and weather features.

Files
- [main.py](main.py): Training script that fits a stacking regressor and serializes `model.pkl` and label encoders.
- [app.py](app.py): Streamlit application to interactively predict yield.
- [india_crop_yield_20000_rows.csv](india_crop_yield_20000_rows.csv): Dataset used for training.
- `model.pkl`, `state.pkl`, `soil.pkl`, `crop.pkl`, `season.pkl`: Pickled artifacts produced by `main.py`.

Requirements
- Python 3.8+ (tested with 3.10)
- Recommended packages:
  - pandas
  - numpy
  - scikit-learn
  - streamlit
  - matplotlib

Quick setup

1. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install pandas numpy scikit-learn streamlit matplotlib
```

Running the app

```bash
# Start the Streamlit app
streamlit run app.py
```

Retrain the model

If you modify the dataset or want to retrain:

```bash
python main.py
```

This will overwrite `model.pkl` and the encoder pickles.

Notes
- `app.py` expects the pickled model and encoders (`model.pkl`, `state.pkl`, `soil.pkl`, `crop.pkl`, `season.pkl`) to exist in the project root.
- The app sends a feature vector of 11 items (encoded categorical features + numeric features). Keep `main.py` and `app.py` consistent when changing features.

Contact
- For questions or improvements, edit the repository or open an issue in your source control system.
