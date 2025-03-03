# Linear Regression with scikit-learn

## Overview
This project demonstrates the implementation of a simple linear regression model using Python and the `scikit-learn` library. The model is trained on a dataset containing house area and prices, then used to make predictions.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Features
- Load data from an Excel file
- Visualize the dataset
- Train a linear regression model
- Predict house prices based on area
- Save predictions to an Excel file

## Installation
Ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

## Usage
1. Place your dataset (`home_price.xlsx`) in the working directory.
2. Run the Jupyter Notebook or Python script to:
   - Load and visualize data
   - Train the model
   - Make predictions
   - Save results to `prediction.xlsx`

## Code Snippet
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load dataset
df = pd.read_excel("home_price.xlsx")

# Train model
reg = linear_model.LinearRegression()
reg.fit(df[['Area']], df['Price'])

# Make prediction
predicted_price = reg.predict([[3300]])
print("Predicted Price:", predicted_price)
```

## Output
- A trained linear regression model
- Graphical representation of data
- Predicted prices for given areas
- `prediction.xlsx` containing predicted results

## License
This project is for educational purposes only. Feel free to modify and use it as needed.

