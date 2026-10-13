import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")

# Inspect the data
print("Dataset shape:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))
print("Duplicate rows:", df.duplicated().sum())

# Calculate a shipping-time variance when these columns exist
if {"Days_for_shipping_real", "Days_for_shipment_scheduled"}.issubset(df.columns):
    df["shipping_variance"] = (
        df["Days_for_shipping_real"] -
        df["Days_for_shipment_scheduled"]
    )
    print("Average shipping-time variance:",
          df["shipping_variance"].mean())

# Explore a pre-existing late-delivery risk field if available
if "Late_delivery_risk" in df.columns:
    print("Late delivery risk distribution:")
    print(df["Late_delivery_risk"].value_counts(normalize=True) * 100)

# Compare late-delivery risk by shipping mode if columns exist
if {"Shipping_Mode", "Late_delivery_risk"}.issubset(df.columns):
    risk_by_mode = df.groupby("Shipping_Mode")[
        "Late_delivery_risk"
    ].mean().sort_values(ascending=False)
    print(risk_by_mode)
    risk_by_mode.plot(kind="bar", title="Late-delivery risk by shipping mode")
    plt.ylabel("Mean risk label")
    plt.tight_layout()
    plt.show()