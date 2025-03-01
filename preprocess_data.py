import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the ODS file
filename = "stromdaten-2024-1mit Preis.ods"

# Read the ODS file, setting the correct header row
df = pd.read_excel(filename, engine="odf", sheet_name=0, header=1, dtype=str)

# Strip spaces and fix potential issues with column names
df.columns = df.columns.str.strip()

# Extract relevant columns (adjust based on actual column positions)
df = df.loc[:, ["Datum", "Stunde", "Smartmeter", "Spotpreis", "Produktion Fronius"]]

# Rename columns for clarity
df.columns = ["date", "hour", "consumption", "price", "pv"]

# Ensure hour is numeric
df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

# Convert `date` column - automatically detects format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Create proper datetime index
df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
df = df.set_index("datetime")

# Convert numeric values and from W to kW
df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["pv"] = pd.to_numeric(df["pv"], errors="coerce")

# Scale from W to kW
df["consumption"] = df["consumption"] / 1000  # Now in kW
df["pv"] = df["pv"] / 1000  # Now in kW

# Convert spot prices from €/MWh to €c/kWh
df["price"] = df["price"] / 1000 * 100  # Now in €c/kWh

# Compute correlation before scaling
correlation_before = df[["consumption", "pv"]].corr()
print("Correlation before scaling:\n", correlation_before)

# Scale PV
# scale = 20
# df["pv"] = df["pv"] * scale
# df["demand"] = df["consumption"] + df["pv"]

# get max pv value
print("max pv value: ", df["pv"].max())

def scale_correlation(df):
    lowest_correlation = {"scale": 0, "correlation": 1}
    for scale in range(20, 30):
        df["pv_scaled"] = df["pv"] * scale
        df["demand"] = df["consumption"] + df["pv_scaled"]

        # Compute correlation after scaling
        correlation_after = df[["demand", "pv_scaled"]].corr()
        print(
            "Correlation for scale ", scale, ":",
            correlation_after["demand"]["pv_scaled"],
        )

        if (
            correlation_after["demand"]["pv_scaled"]
            < lowest_correlation["correlation"]
        ):
            lowest_correlation["correlation"] = correlation_after["demand"][
                "pv_scaled"
            ]
            lowest_correlation["scale"] = scale

    print(f"Lowest correlation for scale {lowest_correlation['scale']}: {lowest_correlation['correlation']}")

    df.drop(columns=["pv_scaled"], inplace=True)
    df["pv"] = df["pv"] * lowest_correlation["scale"]
    df["demand"] = df["consumption"] + df["pv"]
# scale_correlation(df)


def find_best_scale(df, scale_range=(1, 30), step=0.5):
    lowest_correlation = {"scale": 0, "correlation": 1}
    results = []

    for scale in np.arange(scale_range[0], scale_range[1], step):
        df["pv_scaled"] = df["pv"] * scale
        df["demand"] = df["consumption"] + df["pv_scaled"]

        # Compute derivatives
        df["demand_derivative"] = df["demand"].diff().fillna(0)
        df["pv_derivative"] = df["pv_scaled"].diff().fillna(0)

        # Filter only times when PV production is active
        df_filtered = df[df["pv"] > 0]

        # Compute correlation
        correlation = (
            df_filtered[["demand_derivative", "pv_derivative"]].corr().iloc[0, 1]
        )

        print(f"Scale: {scale:.2f}, Correlation: {correlation:.5f}")

        results.append((scale, correlation))

        # Update the best scale found so far
        if abs(correlation) < abs(lowest_correlation["correlation"]):
            lowest_correlation["scale"] = scale
            lowest_correlation["correlation"] = correlation

    print(
        f"\nBest scale found: {lowest_correlation['scale']:.2f} with correlation {lowest_correlation['correlation']:.5f}"
    )

    # Apply the best scale to get the final demand estimate
    df["pv_corrected"] = df["pv"] * lowest_correlation["scale"]
    df["demand_corrected"] = df["consumption"] + df["pv_corrected"]

    return df, lowest_correlation["scale"], results


# Run the function
df, best_scale, results = find_best_scale(df, scale_range=(1, 30), step=0.5)

df["pv"] = df["pv"] * best_scale
df["demand"] = df["consumption"] + df["pv"]


# print("Unique hour values:", df["hour"].unique())
# df = df[df["hour"].between(0, 23)]
# print(f"length of df: {len(df)}")

# print("Duplicate timestamps:", df.index.duplicated().sum())
# df = df[~df.index.duplicated(keep="first")]

# df = df.drop_duplicates(subset=["date", "hour"])

print("duplicate rows: ", df.index.duplicated().sum())
print("length df is: ", len(df))

# Ensure a proper time frequency
df = df.asfreq("h")  # Ensures hourly time steps
# df.index = pd.date_range(start=df.index[0], periods=len(df), freq="H")

# Check missing values statistics
missing_count = df.isna().sum()

# Print results
print("Missing values per column:\n", missing_count)

# Get missing timestamps per column
def get_missing_timestamps(df):    
    missing_timestamps = {col: df[df[col].isna()].index.tolist() for col in df.columns}
    for col, timestamps in missing_timestamps.items():
        print(
            f"Missing timestamps in {col} ({len(timestamps)} missing values):\n", timestamps
        )
        break

get_missing_timestamps(df)

# print("Remaining NaNs before interpolation:\n", df.isna().sum())

# Fill small gaps with interpolation, larger gaps with zero
df.interpolate(limit=10, inplace=True)  # Linear interpolation for up to 6h gaps
# ^above method also fills long gaps partially

# Debugging: Check interpolation effect
# print("Remaining NaNs after interpolation:\n", df.isna().sum())

# get_missing_timestamps(df)

# Identify long gaps *before* filling with zero
# def count_long_gaps(series, threshold=6):
#     is_nan = series.isna()
#     group = (~is_nan).cumsum()  # Assigns a group number to consecutive NaNs
#     gap_sizes = is_nan.groupby(group).sum()  # Counts NaNs per group

#     long_gaps = gap_sizes[gap_sizes > threshold]  # Only gaps >6 hours
#     return long_gaps if not long_gaps.empty else None

# # Apply to all columns
# long_gaps = {col: count_long_gaps(df[col]) for col in df.columns}

# # Print number of long gaps and their sizes
# print("Number of long gaps per column:")
# print({col: len(gaps) for col, gaps in long_gaps.items()})

# print("\nSize of long gaps per column:")
# for col, gaps in long_gaps.items():
#     if not gaps.empty:
#         print(f"{col}:\n", gaps)

df.fillna(0, inplace=True)  # Remaining long gaps set to zero

def negative_demand_days(df):
    # Filter for days where demand becomes negative
    negative_days = df[df["demand"] < 0].index.normalize().unique()

    # Filter the data to only include those days
    df_filtered = df[df.index.normalize().isin(negative_days)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(df_filtered.index, df_filtered["consumption"], label="Consumption", color="blue")
    ax.plot(df_filtered.index, df_filtered["demand"], label="Demand", color="red")
    ax.plot(df_filtered.index, df_filtered["pv"], label="PV Feed-in", color="orange")

    # Add vertical lines at day breaks
    for date in negative_days:
        ax.axvline(date, color="gray", linestyle="--", alpha=0.5)

    # Formatting
    ax.set_title("Days with Negative Demand Values")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power [kW]")
    ax.legend()
    ax.grid()

    plt.show()
negative_demand_days(df)

# Drop redundant columns
df = df[["demand", "price", "pv"]]

if df["demand"].min() < 0:
    sum_negatives = df[df["demand"] < 0]["demand"].sum()
    print("Negative demand detected, sum is:", sum_negatives, " kWh")
elif df["pv"].min() < 0:
    raise ValueError("Negative PV generation detected!")

df["demand"] = df["demand"].clip(lower=0)
df["pv"] = df["pv"].clip(lower=0)

# avoid negative prices for now
# df["price"] = df["price"].clip(lower=0)

print("length df is: ", len(df))

# Save cleaned data for use in oemof
df.to_csv("data.csv")

# print index
print(df.index)

# Display first rows
print(df.head())
