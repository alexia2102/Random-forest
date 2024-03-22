# Random Forest Algorithm with Python

This repository contains an implementation of the Random Forest algorithm using Python, focusing on classification tasks. Random Forest is a versatile machine learning algorithm capable of performing both classification and regression tasks. It operates by constructing a multitude of decision trees during training and outputting the mode of the classes of the individual trees.

## Getting Started

To get started with using the Random Forest algorithm in your project, follow these steps:

1. **Clone the repository:** 
   ```
   git clone https://github.com/yourusername/random-forest-python.git
   ```

2. **Install dependencies:** 
   ```
   pip install -r requirements.txt
   ```

3. **Explore the implementation:** 
   You can explore the implementation of the Random Forest algorithm in the `random_forest.py` file. Additionally, you can find examples of using the algorithm in the `examples` directory.

## Usage

To use the Random Forest algorithm in your own projects, follow these steps:

1. **Import the Random Forest class:**
   ```python
   from random_forest import RandomForestClassifier
   ```

2. **Load your data:**
   ```python
   import pandas as pd

   # Read data from CSV file
   df = pd.read_csv('path/to/your/data.csv')

   # Prepare features (X) and target variable (y)
   X = df[['MFCCs_1', 'MFCCs_2', ..., 'MFCCs_22']]
   y = df['RecordID']
   ```

3. **Split the data into training and testing sets:**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
   ```

4. **Instantiate the Random Forest classifier and fit the model:**
   ```python
   clf = RandomForestClassifier(n_estimators=10, max_samples=0.85, max_features=0.8)
   clf.fit(X_train, y_train)
   ```

5. **Make predictions:**
   ```python
   y_pred = clf.predict(X_test)
   ```

6. **Evaluate the model:**
   ```python
   from sklearn import metrics

   accuracy = metrics.accuracy_score(y_test, y_pred)
   print("Accuracy:", accuracy)
   ```
