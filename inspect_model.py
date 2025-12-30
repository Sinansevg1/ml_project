import joblib
import pandas as pd

model = joblib.load("rf_model.pkl")

print(type(model))
print("\nModel parametreleri:")
for k, v in model.get_params().items():
    print(f"{k}: {v}")


features = ["Voltage", "Current_Import", "Power_Import", "SoC", "Power_Ratio", "SoC_Delta", "Current_to_Voltage", "Power_per_SoC"]

importances = model.feature_importances_

df_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(df_importance)
