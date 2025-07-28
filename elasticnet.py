import joblib
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

TICKER = sys.argv[1].upper()
df = pd.read_csv(f"./data/features/{TICKER}.csv")

def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

features = [
    "garch_vol", "atm_iv", "surprisePercentage", "iv_garch_spread", "breakeven_pct", "estimatedEPS", "estimatedEPS_change", "estimatedEPS_pct_change"
]
df["garch_vol breakeven_pct"] = df["garch_vol"] * df["breakeven_pct"]
df["atm_iv breakeven_pct"] = df["atm_iv"] * df["breakeven_pct"]
df["breakeven_pct estimatedEPS_pct_change"] = df["breakeven_pct"] * df["estimatedEPS_pct_change"]
df["atm_iv estimatedEPS"] = df["atm_iv"] * df["estimatedEPS"]
df["iv_garch_spread estimatedEPS"] = df["iv_garch_spread"] * df["estimatedEPS"]
df["breakeven_pct estimatedEPS"] = df["breakeven_pct"] * df["estimatedEPS"]
# features = [
#     "garch_vol", 
#     "breakeven_pct", 
#     "estimatedEPS", 
#     "garch_vol breakeven_pct",
#     "atm_iv breakeven_pct",
#     "breakeven_pct estimatedEPS_pct_change",
#     "atm_iv estimatedEPS",
#     "iv_garch_spread estimatedEPS",
# ]
# MSE: 0.08365 | R²: 0.079 | Adjusted R²: -0.657  RIDGE MSE: 0.10159 | R²: -0.118 | Adjusted R²: -1.013
features = [
    "garch_vol", 
    "iv_garch_spread",
    "breakeven_pct", 
    "estimatedEPS", 
    "atm_iv estimatedEPS",
]
features = [
    "garch_vol", 
    "breakeven_pct", 
    "atm_iv estimatedEPS",
    ""
]

df = df.dropna(subset=features + ["realized_vol"])

df["reportedDate"] = pd.to_datetime(df["reportedDate"])
train = df[df["reportedDate"].dt.year <= 2022]
test = df[df["reportedDate"].dt.year > 2022]

X_train, y_train = train[features], train["realized_vol"]
X_test, y_test = test[features], test["realized_vol"]

# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# poly.fit(X_train)

# X_train_poly = poly.transform(X_train)
# X_test_poly = poly.transform(X_test)
# feature_names = poly.get_feature_names_out(features)

tscv = TimeSeriesSplit(n_splits=5)

# ElasticNetCV
enet_model = make_pipeline(
    StandardScaler(),
    ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1],
        alphas=np.logspace(-4, 1, 50),
        cv=tscv,
        max_iter=10000000
    )
)
enet_model.fit(X_train, y_train)
enet = enet_model.named_steps["elasticnetcv"]
y_pred_enet = enet_model.predict(X_test)
# Poly
# enet_model.fit(X_train_poly, y_train)
# enet = enet_model.named_steps["elasticnetcv"]
# y_pred_enet = enet_model.predict(X_test_poly)

print("\nElasticNet Performance on Test Set:")
r2_enet = r2_score(y_test, y_pred_enet)
adj_r2_enet = adjusted_r2_score(r2_enet, len(y_test), X_test.shape[1])
print(f"MSE: {mean_squared_error(y_test, y_pred_enet):.5f} | R²: {r2_enet:.3f} | Adjusted R²: {adj_r2_enet:.3f}")
print("ElasticNet Best alpha:", enet.alpha_)
print("ElasticNet Best l1_ratio:", enet.l1_ratio_)
print("\nElasticNet Feature Weights:\n", pd.Series(enet.coef_, index=X_train.columns).sort_values())
# print("\nElasticNet Feature Weights:\n", pd.Series(enet.coef_, index=feature_names).sort_values())

# RidgeCV
ridge_model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-4, 1, 50), cv=tscv)
)
ridge_model.fit(X_train, y_train)
ridge = ridge_model.named_steps["ridgecv"]
y_pred_ridge = ridge_model.predict(X_test)

#Poly
# ridge_model.fit(X_train_poly, y_train)
# ridge = ridge_model.named_steps["ridgecv"]
# y_pred_ridge = ridge_model.predict(X_test_poly)
print("\nRidge Performance on Test Set:")
r2_ridge = r2_score(y_test, y_pred_ridge)
adj_r2_ridge = adjusted_r2_score(r2_ridge, len(y_test), X_test.shape[1])
print(f"MSE: {mean_squared_error(y_test, y_pred_ridge):.5f} | R²: {r2_ridge:.3f} | Adjusted R²: {adj_r2_ridge:.3f}")
print("\nRidge Feature Weights:\n", pd.Series(ridge.coef_, index=X_train.columns).sort_values())
# print("\nRidge Feature Weights:\n", pd.Series(ridge.coef_, index=feature_names).sort_values())

joblib.dump((enet_model, features), "./models/elasticnet_model.pkl")
joblib.dump((ridge_model, features), "./models/ridge_model.pkl")

enet_weights = pd.Series(enet.coef_, index=X_train.columns)
ridge_weights = pd.Series(ridge.coef_, index=X_train.columns)
# enet_weights = pd.Series(enet.coef_, index=feature_names)
# ridge_weights = pd.Series(ridge.coef_, index=feature_names)

enet_importance = enet_weights.abs().sort_values(ascending=False)
ridge_importance = ridge_weights.abs().sort_values(ascending=False)

print("\nElasticNet Feature Importance Ranking:")
print(enet_importance)

print("\nRidge Feature Importance Ranking:")
print(ridge_importance)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

enet_importance.plot(kind='barh', ax=axes[0], title="ElasticNet Importances", color='orange')
ridge_importance.plot(kind='barh', ax=axes[1], title="Ridge Importances", color='skyblue')

for ax in axes:
    ax.set_xlabel("Absolute Coefficient")
    ax.invert_yaxis()
    ax.grid(True)

plt.suptitle("Feature Importance Rankings (Standardized)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("ENET permutation importance")
enet_result = permutation_importance(enet_model, X_test, y_test, n_repeats=30, random_state=0)
sorted_idx = enet_result.importances_mean.argsort()[::-1]
print(pd.Series(enet_result.importances_mean[sorted_idx], index=X_test.columns[sorted_idx]))
# enet_result = permutation_importance(enet_model, X_test_poly, y_test, n_repeats=30, random_state=0)
# sorted_idx = enet_result.importances_mean.argsort()[::-1]
# print(pd.Series(enet_result.importances_mean[sorted_idx], index=feature_names[sorted_idx]))


print("RIDGE permutation importance")
ridge_result = permutation_importance(ridge_model, X_test, y_test, n_repeats=30, random_state=0)
sorted_idx = ridge_result.importances_mean.argsort()[::-1]
print(pd.Series(ridge_result.importances_mean[sorted_idx], index=X_test.columns[sorted_idx]))
# ridge_result = permutation_importance(ridge_model, X_test_poly, y_test, n_repeats=30, random_state=0)
# sorted_idx = ridge_result.importances_mean.argsort()[::-1]
# print(pd.Series(ridge_result.importances_mean[sorted_idx], index=feature_names[sorted_idx]))

# --- Plot ---
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred_ridge, alpha=0.7, label="Ridge")
# plt.scatter(y_test, y_pred_enet, alpha=0.7, label="ElasticNet", marker='x')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual Realized Vol")
# plt.ylabel("Predicted Realized Vol")
# plt.title("Out-of-Sample Realized Vol Prediction")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()