import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz

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

# Create proper datetime index in UTC (since "Datum" column is already UTC-based)
df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

# Explicitly set the timezone to UTC (ensures all timestamps are properly handled)
df["datetime"] = df["datetime"].dt.tz_localize("UTC")

df = df.set_index("datetime")

# show duplicate rows
dup_rows = df.index.duplicated()
if dup_rows.any():
    print("duplicate rows: ", dup_rows.sum())
    print(df[df.index.duplicated()])

# print("df head: ", df.head())

# Convert numeric values and from W to kW
df["consumption"] = (
    pd.to_numeric(df["consumption"], errors="coerce") / 1000
)  # Now in kW
df["price"] = pd.to_numeric(df["price"], errors="coerce") / 1000 * 100  # Now in €c/kWh
df["pv"] = pd.to_numeric(df["pv"], errors="coerce") / 1000  # Now in kW

# Compute correlation before scaling
correlation_before = df[["consumption", "pv"]].corr()
print("Correlation before scaling:\n", correlation_before)


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
# df, best_scale, results = find_best_scale(df, scale_range=(1, 30), step=0.5)
best_scale = 8
df["pv"] = df["pv"] * best_scale
df["demand"] = df["consumption"] + df["pv"]

print("duplicate rows: ", df.index.duplicated().sum())
print("length df is: ", len(df))

# Ensure a proper time frequency
df = df.asfreq("h")  # Ensures hourly time steps

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
# negative_demand_days(df)

# Drop redundant columns
df = df[["demand", "price", "pv"]]

if df["demand"].min() < 0:
    sum_negatives = df[df["demand"] < 0]["demand"].sum()
    print("Negative demand detected, sum is:", sum_negatives, " kWh")
elif df["pv"].min() < 0:
    raise ValueError("Negative PV generation detected!")

df["demand"] = df["demand"].clip(lower=0)
df["pv"] = df["pv"].clip(lower=0)

print("length df is: ", len(df))


def add_cold_price(df):
    # Load the temperature data
    weather_filename = "opsd-weather_data-2020-09-16/weather_data.csv"
    weather_df = pd.read_csv(weather_filename, parse_dates=["utc_timestamp"])

    # Rename and index the weather data
    weather_df = weather_df.rename(columns={"utc_timestamp": "datetime", "DE_temperature": "ambient_temp"})
    weather_df = weather_df[["datetime", "ambient_temp"]]
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], format="%Y-%m-%dT%H%M%SZ")
    weather_df = weather_df.set_index("datetime")

    # Filter for 2016 if available
    weather_df = weather_df[weather_df.index.year == 2016]
    if len(weather_df) == 0:
        raise ValueError("No weather data available for 2016!")

    # Resample to hourly data to match the main dataset
    weather_df = weather_df.resample("h").mean()

    # Shift weather data from 2016 to 2024 (both leap years)
    weather_df.index = weather_df.index + pd.DateOffset(years=8)

    # Merge weather data with main dataset
    df = df.merge(weather_df[["ambient_temp"]], how="left", left_index=True, right_index=True)

    # Define efficiency factor for realistic COP estimation
    EFFICIENCY_FACTOR = 0.4

    # Function to calculate realistic COP
    def calculate_real_cop(T_cold_C=-25, T_hot_C=25, eta=EFFICIENCY_FACTOR):
        T_cold_K = T_cold_C + 273.15  # Convert to Kelvin
        T_hot_K = T_hot_C + 273.15  # Convert to Kelvin
        cop_carnot = T_cold_K / (T_hot_K - T_cold_K)
        return eta * cop_carnot  # Adjust for real-world efficiency

    # Compute COP for each timestamp based on ambient temperature (condenser side + 2°C adjustment)
    df["COP_ambient"] = df["ambient_temp"].apply(lambda T: calculate_real_cop(T_hot_C=T + 2))
    df["cold_temp"] = 3
    df["COP_caes"] = df["cold_temp"].apply(lambda T: calculate_real_cop(T_hot_C=T))

    # Compute cost savings per kWh of cold brine
    df["cold_price"] = (1 / df["COP_ambient"] - 1 / df["COP_caes"]) * df["price"]
    # Set to zero for negative values
    df["cold_price"] = df["cold_price"].clip(lower=0)

    # drop columns
    df = df.drop(columns=["COP_ambient", "cold_temp", "COP_caes"])
    return df

df = add_cold_price(df)

# add heat price
df["heat_price"] = 7 # €c/kWh
# set to zero during summer months
df["heat_price"] = df["heat_price"].mask((df.index.month >= 5) & (df.index.month <= 9), 0)

# plot prices
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df["price"], label="Electricity Price", color="blue")
ax.plot(df["cold_price"], label="Cold Price", color="red")
ax.plot(df["heat_price"], label="Heat Price", color="green")
# ax.plot(df["ambient_temp"], label="Ambient Temperature", color="orange")
# # plot cops
# ax.plot(df["COP_ambient"], label="COP Ambient", color="purple")
# ax.plot(df["COP_caes"], label="COP CAES", color="black")
ax.set_title("Electricity, Cold and Heat Prices")
ax.set_xlabel("Time")
ax.set_ylabel("Price [€c/kWh]")
ax.legend()
ax.grid()
plt.show()

# Save cleaned data for use in oemof
df.to_csv("data.csv")

# print index
print(df.index)

# Display first rows
print(df.head())
