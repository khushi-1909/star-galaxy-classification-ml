# Star-Galaxy Classification Using Machine Learning

This project applies machine learning techniques to classify celestial objects (stars vs galaxies) using photometric data from the Sloan Digital Sky Survey (SDSS).

## Features
- Feature engineering using color indices and PCA
- Multiple ML classifiers (Logistic Regression, Random Forest, XGBoost)
- Evaluation using AUC-ROC and F1-score
- Data visualization and model comparison

## Folder Structure
- `data/`: Raw and cleaned SDSS photometric data
- `notebooks/`: Jupyter Notebooks for exploration and modeling
- `src/`: Core scripts for preprocessing and training
- `outputs/`: Graphs and metrics
- `models/`: Trained model files (optional)

## Getting Started
```bash
pip install -r requirements.txt
python src/train_model.py
```

## Author
Khushi Verma

## License
MIT
