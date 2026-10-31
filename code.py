

# PROJECT: CUSTOMER RETENTION AND REVENUE INTELLIGENCE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, classification_report,ConfusionMatrixDisplay)

# Create output folder
output_folder = Path("Customer_Retention_Results")
output_folder.mkdir(exist_ok=True)

#DATA PREPROCESSING AND CLEANING

print("=" * 60)
print("CUSTOMER RETENTION AND REVENUE INTELLIGENCE")
print("=" * 60)

file_path = r"C:\Users\adija\OneDrive\Desktop\INT-234\online_retail_II.xlsx"

# Load both sheets
sheets = pd.read_excel(file_path, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

df.columns = df.columns.str.strip()

print("\nDATASET LOADED SUCCESSFULLY")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Column names:", df.columns.tolist())

# Inspect data
print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

print("\nDUPLICATE ROWS BEFORE CLEANING")
print(df.duplicated().sum())

# Create cleaned copy
clean_df = df.copy()

# Remove duplicates
clean_df = clean_df.drop_duplicates()

# Convert columns to correct data types
clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"], errors="coerce")

clean_df["Quantity"] = pd.to_numeric(clean_df["Quantity"], errors="coerce")

clean_df["Price"] = pd.to_numeric(clean_df["Price"], errors="coerce")

# Remove rows missing essential fields
clean_df = clean_df.dropna(subset=["Invoice", "InvoiceDate", "Quantity", "Price"])

# Remove zero quantity and invalid prices
clean_df = clean_df[(clean_df["Quantity"] != 0) &(clean_df["Price"] > 0)].copy()

# Handle missing descriptions
clean_df["Description"] = clean_df["Description"].fillna("Unknown Product")

clean_df["Description"] = (clean_df["Description"].astype(str).str.strip())

clean_df.loc[clean_df["Description"] == "","Description"
] = "Unknown Product"

# Standardize invoice numbers
clean_df["Invoice"] = (
    clean_df["Invoice"].astype(str).str.strip()
)

# Identify cancellations and possible returns
clean_df["IsCancelled"] = (clean_df["Invoice"].str.upper().str.startswith("C"))

clean_df["IsReturn"] = clean_df["Quantity"] < 0

# Feature engineering: calculate revenue
clean_df["Revenue"] = (clean_df["Quantity"] * clean_df["Price"])

print("\nDATA CLEANING RESULTS")
print("Rows after cleaning:", len(clean_df))
print("Duplicates remaining:", clean_df.duplicated().sum())
print("Missing descriptions:", clean_df["Description"].isnull().sum())
print("Missing customer IDs:", clean_df["Customer ID"].isnull().sum())
print("Cancellation rows:", clean_df["IsCancelled"].sum())
print("Possible return rows:", clean_df["IsReturn"].sum())

clean_df.to_csv(output_folder / "cleaned_retail_data.csv",index=False)

# Prepare valid sales for analysis
sales_df = clean_df[(~clean_df["IsCancelled"]) & (clean_df["Quantity"] > 0) &(clean_df["Price"] > 0)].copy()

sales_df["Revenue"] = (sales_df["Quantity"] * sales_df["Price"])

sales_df["YearMonth"] = (sales_df["InvoiceDate"].dt.to_period("M").astype(str))

print("\nSALES OVERVIEW")
print("Valid sales rows:", len(sales_df))
print("Total revenue:", round(sales_df["Revenue"].sum(), 2))
print("Unique invoices:", sales_df["Invoice"].nunique())
print("Unique products:", sales_df["StockCode"].nunique())
print("Unique countries:", sales_df["Country"].nunique())

#EDA, OBJECTIVES AND VISUALIZATION

print("\nPROJECT OBJECTIVES")
print("1. Analyze monthly revenue trends.")
print("2. Identify top products by revenue.")
print("3. Identify top countries by revenue.")
print("4. Analyze customer purchasing behavior.")
print("5. Segment customers using RFM features.")
print("6. Predict repeat purchases within 90 days.")

print("\nSTATISTICAL SUMMARY")
print(sales_df[["Quantity", "Price", "Revenue"]].describe())

# OBJECTIVE 1: Monthly revenue trends
monthly_revenue = (sales_df.groupby("YearMonth")["Revenue"].sum().sort_index())

print("\nMONTHLY REVENUE")
print(monthly_revenue)

monthly_revenue.plot(marker="o", figsize=(12, 5))
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_folder / "01_monthly_revenue.png")
plt.close()

# OBJECTIVE 2: Top products
product_revenue = (
    sales_df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP 10 PRODUCTS")
print(product_revenue)

product_revenue.sort_values().plot(
    kind="barh", figsize=(10, 6)
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(output_folder / "02_top_products.png")
plt.close()

# OBJECTIVE 3: Top countries
country_revenue = (
    sales_df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP 10 COUNTRIES")
print(country_revenue)

country_revenue.sort_values().plot(
    kind="barh", figsize=(10, 6)
)

plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(output_folder / "03_top_countries.png")
plt.close()

# OBJECTIVE 4: Customer purchasing behavior
customer_sales = sales_df.dropna(
    subset=["Customer ID"]
).copy()

customer_sales["Customer ID"] = (
    customer_sales["Customer ID"].astype(str)
)

customer_sales = customer_sales[
    customer_sales["Customer ID"].str.strip() != ""
].copy()

customer_summary = customer_sales.groupby("Customer ID").agg(
    TotalRevenue=("Revenue", "sum"),
    NumberOfInvoices=("Invoice", "nunique"),
    TotalQuantity=("Quantity", "sum"),
    UniqueProducts=("StockCode", "nunique")
)

customer_summary = customer_summary.sort_values(
    "TotalRevenue", ascending=False
)

print("\nTOP 10 CUSTOMERS")
print(customer_summary.head(10))

customer_summary.head(10)["TotalRevenue"].sort_values().plot(
    kind="barh", figsize=(10, 5)
)

plt.title("Top 10 Customers by Revenue")
plt.xlabel("Revenue")
plt.tight_layout()
plt.savefig(output_folder / "04_top_customers.png")
plt.close()

# Additional visualization: revenue distribution
sns.histplot(
    sales_df["Revenue"].clip(
        upper=sales_df["Revenue"].quantile(0.99)
    ),
    bins=40
)

plt.title("Transaction Revenue Distribution")
plt.xlabel("Revenue (capped at 99th percentile)")
plt.tight_layout()
plt.savefig(output_folder / "05_revenue_distribution.png")
plt.close()

sales_df.to_csv(
    output_folder / "valid_sales_data.csv",
    index=False
)


# ============================================================
# REQUIREMENT 5: FEATURE ENGINEERING AND RFM CLUSTERING
# ============================================================

print("\nRFM CUSTOMER SEGMENTATION")

reference_date = (
    customer_sales["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)

rfm = customer_sales.groupby("Customer ID").agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum"),
    TotalQuantity=("Quantity", "sum"),
    UniqueProducts=("StockCode", "nunique")
)

# Recency: days since last purchase
rfm["Recency"] = (
    reference_date - rfm["LastPurchase"]
).dt.days

# Average order value
rfm["AverageOrderValue"] = (
    rfm["Monetary"] / rfm["Frequency"]
)

print("\nRFM SUMMARY")
print(
    rfm[
        ["Recency", "Frequency", "Monetary", "AverageOrderValue"]
    ].describe()
)

# Prepare features for clustering
cluster_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AverageOrderValue"
]

cluster_data = np.log1p(
    rfm[cluster_columns].clip(lower=0)
)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Group customers into four segments
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Segment"] = kmeans.fit_predict(scaled_data)

segment_summary = rfm.groupby("Segment").agg(
    Customers=("Recency", "count"),
    AverageRecency=("Recency", "mean"),
    AverageFrequency=("Frequency", "mean"),
    AverageMonetary=("Monetary", "mean")
)

print("\nCUSTOMER SEGMENT SUMMARY")
print(segment_summary.round(2))

segment_summary["Customers"].plot(
    kind="bar", figsize=(8, 5)
)

plt.title("Customer Count by Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(output_folder / "06_customer_segments.png")
plt.close()

rfm.to_csv(output_folder / "customer_segments.csv")
segment_summary.to_csv(output_folder / "segment_summary.csv")

#FOUR MODELS AND EVALUATION


print("\nREPEAT-PURCHASE PREDICTION")

latest_date = customer_sales["InvoiceDate"].max()
cutoff_date = latest_date - pd.Timedelta(days=90)

# Historical data used for features
historical_sales = customer_sales[
    customer_sales["InvoiceDate"] < cutoff_date
].copy()

# Final 90 days used to create target labels
future_sales = customer_sales[
    (customer_sales["InvoiceDate"] >= cutoff_date) &
    (customer_sales["InvoiceDate"] <= latest_date)
].copy()

features = historical_sales.groupby("Customer ID").agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum"),
    TotalQuantity=("Quantity", "sum"),
    UniqueProducts=("StockCode", "nunique"),
    UniqueCountries=("Country", "nunique")
)

features["Recency"] = (
    cutoff_date - features["LastPurchase"]
).dt.days

features["AverageOrderValue"] = (
    features["Monetary"] / features["Frequency"]
)

future_customers = set(
    future_sales["Customer ID"].astype(str).unique()
)

features["RepeatPurchase"] = (
    features.index.astype(str).isin(future_customers).astype(int)
)

features = features.drop(columns=["LastPurchase"])
features = features.replace([np.inf, -np.inf], np.nan)

feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "UniqueCountries",
    "AverageOrderValue"
]

X = features[feature_columns]
y = features["RepeatPurchase"]

print("Model dataset shape:", X.shape)
print("Target counts:")
print(y.value_counts())

if len(features) < 20 or y.nunique() < 2:
    print("Not enough data or target classes to train the models.")
else:
    # Divide customers into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Four machine-learning models
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                random_state=42
            ))
        ]),

        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        print("\nTraining:", name)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        auc = roc_auc_score(y_test, probabilities)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": auc
        })

        trained_models[name] = model

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))
        print("ROC-AUC:", round(auc, 4))

    # Compare models
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "F1 Score", ascending=False
    )

    print("\nMODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    results_df.to_csv(
        output_folder / "model_comparison.csv",
        index=False
    )

    # Select best model based on F1-score
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    best_predictions = best_model.predict(X_test)

    print("\nBEST MODEL:", best_name)
    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test, best_predictions, zero_division=0
    ))

    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_predictions
    )

    plt.title("Confusion Matrix - " + best_name)
    plt.tight_layout()
    plt.savefig(output_folder / "07_confusion_matrix.png")
    plt.close()

    # Visual comparison of model metrics
    results_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ].plot(kind="bar", figsize=(12, 6))

    plt.title("Comparison of Four Machine-Learning Models")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_folder / "08_model_comparison.png")
    plt.close()

    # Feature importance if the best model supports it
    best_estimator = best_model.named_steps["model"]

    if hasattr(best_estimator, "feature_importances_"):
        importance = pd.Series(
            best_estimator.feature_importances_,
            index=feature_columns
        ).sort_values(ascending=False)

        print("\nFEATURE IMPORTANCE")
        print(importance.round(4))

        importance.sort_values().plot(
            kind="barh", figsize=(9, 5)
        )

        plt.title("Feature Importance - " + best_name)
        plt.tight_layout()
        plt.savefig(output_folder / "09_feature_importance.png")
        plt.close()

    features.to_csv(
        output_folder / "customer_prediction_features.csv"
    )










