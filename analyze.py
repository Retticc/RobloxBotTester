import sys, os
import pandas as pd

def quality_report(df):
    report = []
    report.append(f"Total records: {len(df)}")
    for col in df.columns:
        missing = df[col].isna().sum()
        pct = missing / len(df) * 100
        report.append(f"- {col}: {missing} missing ({pct:.1f}%)")
    return "\n".join(report)

def feasibility_report(df, delay):
    rate = 60 / delay if delay else 0
    return (
        f"Rate-limit check:\n"
        f"- Records fetched: {len(df)}\n"
        f"- Delay per request: {delay}s\n"
        f"- Approx req/min: {rate:.1f}\n"
        f"- Sustainable: {'Yes' if rate <= 100 else 'No'}"
    )

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "test_data.csv"
    df = pd.read_csv(path)
    delay = float(os.getenv("RATE_LIMIT_DELAY", "0.7"))
    print("=== Data Quality Report ===")
    print(quality_report(df))
    print("\n=== Feasibility Report ===")
    print(feasibility_report(df, delay))
