# House Price Prediction

## Project Overview

House Price Prediction is a Machine Learning project that estimates the selling price of a house based on various property features such as area, number of rooms, floor, location, and house age. The project uses regression algorithms to analyze historical housing data and predict accurate house prices.

The objective of this project is to assist buyers, sellers, and real estate professionals by providing reliable house price predictions based on property characteristics.

---

## Features

- Data preprocessing and cleaning
- Missing value handling
- Duplicate record removal
- Feature engineering
- Categorical feature encoding
- Feature scaling
- Training multiple regression models
- Model evaluation using regression metrics
- House price prediction through a Streamlit web application
- Model saving and loading using Joblib

---

## Dataset

The dataset contains information about residential properties, including:

- Area
- Number of Rooms
- Location
- Floor
- Year Built
- House Age
- Price (Target Variable)

Target Variable:

- Price

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate records
- Handled missing values
- Converted data into appropriate formats
- Encoded categorical variables
- Created a new feature called House Age
- Standardized numerical features
- Split the dataset into training and testing sets

---

## Feature Engineering

The project includes feature engineering techniques to improve prediction accuracy.

Examples include:

- House Age = Current Year - Year Built
- Encoded location values
- Selected important features for training

---

## Machine Learning Models

The following regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best-performing model was selected based on evaluation metrics.

---

## Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Streamlit Application

The project includes a Streamlit application that allows users to:

- Enter house details
- Predict house prices instantly
- Display the estimated selling price

Run the application using:

```bash
streamlit run app.py
```

---

## Project Structure

```
House-Price-Prediction/
│
├── dataset/
│   └── house_data.csv
│
├── models/
│   ├── house_price_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
```

Navigate to the project folder:

```bash
cd House-Price-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Workflow

1. Load the house price dataset.
2. Perform data preprocessing and cleaning.
3. Apply feature engineering.
4. Train multiple regression models.
5. Evaluate model performance.
6. Save the best-performing model.
7. Load the trained model into the Streamlit application.
8. Enter house details to predict the estimated price.

---

## Future Enhancements

- Improve prediction accuracy using advanced ensemble models.
- Deploy the application on cloud platforms.
- Add interactive visualizations and dashboards.
- Integrate real-time property data.
- Support batch predictions using CSV file uploads.

## Output

<img width="1913" height="908" alt="image" src="https://github.com/user-attachments/assets/cb7c92c6-ab94-4ed6-9aaf-11bbf1334ce3" />


<img width="1910" height="917" alt="image" src="https://github.com/user-attachments/assets/f8b88ea6-88c3-43bd-9aaf-0b5a42517df8" />
