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
# print price average
print("price average: ", df["price"].mean())
df["pv"] = pd.to_numeric(df["pv"], errors="coerce") / 1000  # Now in kW

def calc_price(df):
    scale_price = 1
    scale_grid = 1
    scale_taxes = 1
    add_on_price = 7.53 * scale_grid  # €cent/kWh  7.53 zu 13.08
    MwSt = 0.19 * scale_taxes
    df["price"] = df["price"] * scale_price + add_on_price
    df["price"] = df["price"] * (1 + MwSt)
    return df

df = calc_price(df)
print("price average: ", df["price"].mean())
# print hours below 8.55 €cent/kWh
hours_below = len(df[df["price"] < 8.55])
print("hours below 8.55 €cent/kWh: ", hours_below, " hours. And ", hours_below / len(df) * 100, " %")

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
# df, best_scale, results = find_best_scale(df, scale_range=(1, 30), step=0.2)
# best_scale = 8
# the sum of pv feed-in should be 119.669 kWh
best_scale = 119_669 / df["pv"].sum()
print("best scale is: ", best_scale)
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

    # plt.show()
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

print("length df is: ", len(df))


def plot_cop_values(df):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df["COP_ambient_freezer"], label="COP Ambient Freezer", color="blue")
    ax.plot(df.index, df["COP_ambient_chiller"], label="COP Ambient Chiller", color="cyan")
    ax.plot(df.index, df["COP_caes_freezer"], label="COP CAES Freezer", color="purple")
    ax.plot(df.index, df["COP_caes_chiller"], label="COP CAES Chiller", color="orange")
    ax.set_title("Coefficient of Performance (COP) Values")
    ax.set_xlabel("Time")
    ax.set_ylabel("COP")
    ax.legend()
    ax.grid()


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
    # def get_eta_for_target_cop(df, T_cold_C, column_name, target_mean_cop):
    #     T_hot_K = df["ambient_temp"] + 2 + 273.15
    #     T_cold_K = T_cold_C + 273.15
    #     delta_T = T_hot_K - T_cold_K
    #     # Vermeide Division durch Null
    #     delta_T = delta_T.clip(lower=0.1)
    #     cop_carnot = T_cold_K / delta_T
    #     eta = target_mean_cop / cop_carnot.mean()
    #     return eta

    # eta_nk = get_eta_for_target_cop(
    #     df, T_cold_C=0, column_name="ambient_temp", target_mean_cop=2.8
    # )
    # eta_tk = get_eta_for_target_cop(
    #     df, T_cold_C=-27, column_name="ambient_temp", target_mean_cop=1.2
    # )
    # print("eta_nk: ", eta_nk)
    # print("eta_tk: ", eta_tk)

    # Function to calculate realistic COP
    def calculate_real_cop(T_cold_C=-25, T_hot_C=25, eta=1):
        T_cold_K = T_cold_C + 273.15  # Convert to Kelvin
        T_hot_K = T_hot_C + 273.15  # Convert to Kelvin
        if T_cold_K > T_hot_K - 2:
            T_hot_K = T_cold_K + 2  # Avoid division by zero
        cop_carnot = T_cold_K / (T_hot_K - T_cold_K)
        return eta * cop_carnot  # Adjust for real-world efficiency

    COP_MAX = 18
    T_delta_air = 12  # °C temperature difference between air and condensor side
    T_delta = 10  # °C temperature difference between brine and condensor side

    # Compute COP for each timestamp based on ambient temperature (condenser side + 2°C adjustment)
    df["COP_ambient_freezer"] = df["ambient_temp"].apply(
        lambda T: calculate_real_cop(T_cold_C=-27, T_hot_C=T + T_delta_air)
    )
    COP_ambient_freezer_expected = 1.2  # Expected mean COP for freezer
    COP_ambient_freezer_mean = df["COP_ambient_freezer"].mean()
    efficiency_factor_freezer = COP_ambient_freezer_expected / COP_ambient_freezer_mean
    df["COP_ambient_freezer"] = df["COP_ambient_freezer"] * efficiency_factor_freezer
    print("efficiency factor freezer: ", efficiency_factor_freezer)

    df["COP_ambient_chiller"] = (
        df["ambient_temp"]
        .apply(lambda T: calculate_real_cop(T_cold_C=0, T_hot_C=T + T_delta_air))
        .clip(upper=COP_MAX)
    )
    COP_ambient_chiller_expected = 2.8  # Expected mean COP for chiller
    COP_ambient_chiller_mean = df["COP_ambient_chiller"].mean()
    efficiency_factor_chiller = COP_ambient_chiller_expected / COP_ambient_chiller_mean
    df["COP_ambient_chiller"] = df["COP_ambient_chiller"] * efficiency_factor_chiller
    print("efficiency factor chiller: ", efficiency_factor_chiller)

    df["cold_temp"] = 3  # °C brine temperature (on condenser side)
    df["COP_caes_freezer"] = df["cold_temp"].apply(lambda T: calculate_real_cop(T_cold_C=-27, T_hot_C=T + T_delta, eta=efficiency_factor_freezer))
    df["COP_caes_chiller"] = (
        df["cold_temp"]
        .apply(
            lambda T: calculate_real_cop(
                T_cold_C=0, T_hot_C=T + T_delta, eta=efficiency_factor_chiller
            )
        )
        .clip(upper=COP_MAX * efficiency_factor_chiller)
    )

    # Compute cost savings per kWh of cold brine
    df["cold_price_freezer"] = (1 / df["COP_ambient_freezer"] - 1 / df["COP_caes_freezer"]) * df["price"]
    df["cold_price_chiller"] = (1 / df["COP_ambient_chiller"] - 1 / df["COP_caes_chiller"]) * df["price"]
    # Set to zero for negative values
    df["cold_price_freezer"] = df["cold_price_freezer"].clip(lower=0)
    df["cold_price_chiller"] = df["cold_price_chiller"].clip(lower=0)

    # drop columns
    # df = df.drop(columns=["COP_ambient_freezer", "COP_ambient_chiller", "cold_temp", "COP_caes_freezer", "COP_caes_chiller"])
    plot_cop_values(df)
    return df
df = add_cold_price(df)


def calculate_basement_temp(df):
    T_ground = 10  # °C
    T_ambient = df["ambient_temp"]
    w = 0.5  # weight
    df["T_basement"] = w * T_ground + (1 - w) * df["ambient_temp"]
    # df["T_basement"] = df["T_basement"].clip(lower=0)
    return df
df = calculate_basement_temp(df)


def calculate_cooling_demand(df):
    T_chiller = 5  # °C
    T_freezer = -18  # °C
    df["T_diff_chiller"] = df["T_basement"] - T_chiller  # °C
    df["T_diff_chiller"] = df["T_diff_chiller"].clip(lower=0)
    df["T_diff_freezer"] = df["T_basement"] - T_freezer  # °C
    df["T_diff_freezer"] = df["T_diff_freezer"].clip(lower=0)
    Qyear_chiller = 25_620  # kWh/a
    Qyear_freezer = 14_230  # kWh/a
    # heat demand per hour is
    # Phi = UA * deltaT
    # We know deltaT, but we don't know UA
    # So we can calculate UA from the yearly demand
    yearly_diff_chiller = df["T_diff_chiller"].sum()
    # UA = Qyear / yearly_diff
    UA_chiller = Qyear_chiller / yearly_diff_chiller
    df["Q_demand_chiller"] = UA_chiller * df["T_diff_chiller"]

    yearly_diff_freezer = df["T_diff_freezer"].sum()
    UA_freezer = Qyear_freezer / yearly_diff_freezer
    df["Q_demand_freezer"] = UA_freezer * df["T_diff_freezer"]

    print("UA_chiller: ", UA_chiller)
    print("UA_freezer: ", UA_freezer)
calculate_cooling_demand(df)


def plot_temperature_and_demand(df):
    """
    Plots the basement temperature and the cooling demand on two vertical axes.
    The left vertical axis shows temperatures and the right vertical axis shows cooling demand (kWh/h).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot temperatures on the left axis
    ax1.plot(df.index, df["T_basement"], label="Basement Temperature", color="blue")
    ax1.axhline(5, color="green", linestyle="--", label="chiller Setpoint (5°C)")
    ax1.axhline(-18, color="red", linestyle="--", label="freezer Setpoint (-18°C)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature (°C)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create a second vertical axis for the cooling demand
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["Q_demand_chiller"], label="Cooling Demand chiller", color="orange")
    ax2.plot(df.index, df["Q_demand_freezer"], label="Cooling Demand freezer", color="purple")
    ax2.set_ylabel("Cooling Demand (kWh/h)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Basement Temperature and Cooling Demand")
    plt.tight_layout()
    # plt.show()


# Example usage:
# Assuming 'df' is your DataFrame with a DateTimeIndex and already contains the calculated fields.
plot_temperature_and_demand(df)


# add heat price
heat_price = 9.5 # €cent/kWh
df["heat_price"] = heat_price
# set to zero during summer months
df["heat_price"] = df["heat_price"].mask((df.index.month >= 5) & (df.index.month <= 9), heat_price * 0.5)

# plot prices
fig, ax = plt.subplots(figsize=(15, 5))
# ax.plot(df["price"], label="Electricity Price", color="blue")
ax.plot(df["cold_price_freezer"], label="Cold Price freezer", color="blue")
ax.plot(df["cold_price_chiller"], label="Cold Price chiller", color="cyan")
ax.plot(df["heat_price"], label="Heat Price", color="red")
# ax.plot(df["ambient_temp"], label="Ambient Temperature", color="orange")
# # plot cops
# ax.plot(df["COP_ambient"], label="COP Ambient", color="purple")
# ax.plot(df["COP_caes"], label="COP CAES", color="black")
ax.set_title("Electricity, Cold and Heat Prices")
ax.set_xlabel("Time")
ax.set_ylabel("Price [€c/kWh]")
ax.legend()
ax.grid()
# plt.show()

# Save cleaned data for use in oemof
df.to_csv("data.csv")

# print index
# print(df.index)

# print max pv production
print("max pv production: ", df["pv"].max())

# Display first rows
print(df.head())
plt.show()
