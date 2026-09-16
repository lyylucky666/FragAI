# Machine Learning Multi-Model Classification and Prediction

This project is a Python-based machine learning classification project. It uses the dataset in `data/data.xlsx` to train, evaluate, and compare multiple classification models. 

## Features

- Load the dataset from `data/data.xlsx`
- Train and evaluate multiple machine learning classification models
- Compare model performance and save the comparison results
- Provide `predict.ipynb` for loading saved models and making predictions

## Project Structure

```text
.
├── .gitignore
├── requirements.txt
├── data/
│   └── data.xlsx
├── figures/
│   ├── confusion_matrix_test_decision_tree.png
│   ├── confusion_matrix_test_gradient_boosting.png
│   ├── confusion_matrix_test_mlp.png
│   ├── confusion_matrix_test_random_forest.png
│   ├── confusion_matrix_train_decision_tree.png
│   ├── confusion_matrix_train_gradient_boosting.png
│   ├── confusion_matrix_train_mlp.png
│   ├── confusion_matrix_train_random_forest.png
│   └── figure_1_model_comparison.tif
├── models/
│   ├── best_model.pkl
│   ├── Decision Tree.pkl
│   ├── Gradient Boosting.pkl
│   ├── MLP.pkl
│   └── Random Forest.pkl
├── results/
│   └── model_comparison_results.csv
└── src/
    ├── main.py
    └── predict.ipynb
```


## Models Used

This project includes the following models:

- Decision Tree
- Random Forest
- Gradient Boosting
- MLP

After training, the script selects the best-performing model based on evaluation metrics and saves it as `models/best_model.pkl`.


## Installation

Run the following command in the project root directory:

```bash
pip install -r requirements.txt
```

If you prefer using a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```


