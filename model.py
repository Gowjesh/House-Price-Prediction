## PROBLEM STATEMENT
# House Price Prediction

## Import Library
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


## Load Dataset
pf = pd.read_csv("./house_data.csv")


## EDA
print(pf.head())
print(pf.info())
print(pf.describe())


## Handle Missing Values
pf.dropna(inplace=True)


## Outlier Treatment
Q1 = pf["Price"].quantile(0.25)
Q3 = pf["Price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

pf = pf[(pf["Price"] >= lower) & (pf["Price"] <= upper)]


## Feature Engineering

pf["house_age"] = 2026 - pf["Year"]

pf.drop("Year", axis=1, inplace=True)


## Encoding

pf = pd.get_dummies(
    pf,
    columns=["Location"],
    drop_first=True
)

# Remove invalid characters
pf.columns = (
    pf.columns.astype(str).str.replace(r"[\[\]<>]", "", regex=True)
)

## Train Test Split

X = pf.drop("Price", axis=1)
y = pf["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


## Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


## Model Building

lr = LinearRegression()

ridge = Ridge(alpha=0.1)

lasso = Lasso(
    alpha=0.1,
    max_iter=10000
)

elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    max_iter=10000
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)


## Model Training

# Models requiring scaling

lr.fit(X_train_scaled, y_train)

ridge.fit(X_train_scaled, y_train)

lasso.fit(X_train_scaled, y_train)

elastic.fit(X_train_scaled, y_train)

# Tree Models

rf.fit(X_train, y_train)

xgb.fit(X_train, y_train)


## Model Prediction

lr_pred = lr.predict(X_test_scaled)

ridge_pred = ridge.predict(X_test_scaled)

lasso_pred = lasso.predict(X_test_scaled)

elastic_pred = elastic.predict(X_test_scaled)

rf_pred = rf.predict(X_test)

xgb_pred = xgb.predict(X_test)

## Model Evaluation

def evaluate_model(model, X, y, y_test, y_pred, n, p):

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    cv_score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    ).mean()

    return mae, mse, rmse, r2, adj_r2, cv_score


## Model Comparison

models = {

    "Linear Regression": (lr, lr_pred),

    "Ridge Regression": (ridge, ridge_pred),

    "Lasso Regression": (lasso, lasso_pred),

    "Elastic Net": (elastic, elastic_pred),

    "Random Forest": (rf, rf_pred),

    "XGBoost": (xgb, xgb_pred)

}


n = X_test.shape[0]

p = X_test.shape[1]

results = []


for name, (model, pred) in models.items():

    mae, mse, rmse, r2, adj_r2, cv_score = evaluate_model(
        model,
        X,
        y,
        y_test,
        pred,
        n,
        p
    )

    results.append([
        name,
        mae,
        mse,
        rmse,
        r2,
        adj_r2,
        cv_score
    ])


result_pf = pd.DataFrame(

    results,

    columns=[

        "Model",

        "MAE",

        "MSE",

        "RMSE",

        "R2_Score",

        "Adjusted_R2_Score",

        "Cross_Val_Score"

    ]

)


result_pf = result_pf.sort_values(

    by="R2_Score",

    ascending=False

)

print(result_pf)


## Best Model

best_model_name = result_pf.iloc[0]["Model"]

print("\nBest Model :", best_model_name)


if best_model_name == "Linear Regression":

    best_model = lr

elif best_model_name == "Ridge Regression":

    best_model = ridge

elif best_model_name == "Lasso Regression":

    best_model = lasso

elif best_model_name == "Elastic Net":

    best_model = elastic

elif best_model_name == "Random Forest":

    best_model = rf

else:

    best_model = xgb


## Save Model

joblib.dump(best_model, "house_price_model.pkl")


## Save Feature Columns

joblib.dump(X.columns.tolist(), "columns.pkl")


## Save Scaler
# Only required for Linear Models

if best_model_name not in ["Random Forest", "XGBoost"]:

    joblib.dump(scaler, "scaler.pkl")


print("\nModel Saved Successfully...")