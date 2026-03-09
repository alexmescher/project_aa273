from src.preprocessing import load_gpx, summarize_gpx, load_csv

def main():
    df = load_gpx("data/04-Mar-2026-1323.gpx")
    print(df.head())
    df2 = load_csv("data/imu_20260304_132152.csv")
    print(df2.head())

    summary = summarize_gpx(df)
    print(summary)

if __name__ == "__main__":
    main()