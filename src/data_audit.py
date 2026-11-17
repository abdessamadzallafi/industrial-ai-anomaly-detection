import pandas as pd

DATA_PATH = "data/raw/ai4i2020.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

report = []

report.append("AI4I 2020 DATASET AUDIT")
report.append("=" * 45)

report.append(f"Rows: {len(df)}")
report.append(f"Columns: {len(df.columns)}")
report.append(f"Missing values: {df.isnull().sum().sum()}")
report.append(f"Duplicate rows: {df.duplicated().sum()}")

report.append("\nMACHINE FAILURE DISTRIBUTION")
report.append(str(df["Machine failure"].value_counts()))
report.append(
    f"Failure rate: {df['Machine failure'].mean() * 100:.2f}%"
)

report.append("\nFAILURE MODES")
report.append(str(df[failure_columns].sum()))

report.append("\nSENSOR STATISTICS")
report.append(str(df[sensor_columns].describe().round(2)))

report_text = "\n".join(report)

print(report_text)

with open("results/data_audit.txt", "w") as file:
    file.write(report_text)

print("\nAudit saved to results/data_audit.txt")
