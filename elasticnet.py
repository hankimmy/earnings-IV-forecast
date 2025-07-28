import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("./data/features/AAPL.csv")

features = [
    "garch_vol", "atm_iv", "estimatedEPS", "surprise", "surprisePercentage"
]
df = df.dropna(subset=features + ["realized_vol"])

X = df[features]
y = df["realized_vol"]
model = make_pipeline(
    StandardScaler(),
    ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1],
        alphas=np.logspace(-4, 1, 50),
        cv=TimeSeriesSplit(n_splits=5)
    )
)
model.fit(X, y)

enet = model.named_steps["elasticnetcv"]
print("Best alpha:", enet.alpha_)
print("Best l1_ratio:", enet.l1_ratio_)

coef = pd.Series(enet.coef_, index=X.columns)
print("\nFeature Weights:\n", coef.sort_values())

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\nMSE: {mse:.5f} | R²: {r2:.3f}")

joblib.dump((model, features), "./models/elasticnet_model.pkl")

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Realized Vol")
plt.ylabel("Predicted Realized Vol")
plt.title("ElasticNetCV: Realized Vol Prediction")
plt.grid(True)
plt.tight_layout()
plt.show()
