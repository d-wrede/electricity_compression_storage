import pandas as pd
import oemof.solph as solph
import matplotlib.pyplot as plt

# Define a constant feed-in price in €/Wh
feed_in_price = 14  # €cent/kWh


def get_data():
    # Load time series data (assuming you have a CSV file with datetime index)
    df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

    # Ensure DatetimeIndex is sorted
    df = df.sort_index()

    # Compute expected index with the correct number of periods
    expected_index = pd.date_range(start=df.index.min(), periods=len(df), freq="h")

    # Check for missing timestamps
    missing_timestamps = expected_index.difference(df.index)

    if not missing_timestamps.empty:
        print(f"Warning: {len(missing_timestamps)} timestamps are missing!")
        print("Missing timestamps:", missing_timestamps.tolist())

    # Reindex to ensure all expected timestamps are present, filling gaps with NaN
    df = df.reindex(expected_index)

    # Restore hourly frequency
    df = df.asfreq("h")

    # Handle missing values (decide based on your use case)
    df.interpolate(limit_direction="both", inplace=True)  # Fill gaps with interpolation
    df.fillna(0, inplace=True)  # If long gaps exist, replace remaining NaNs with zero

    # Confirm frequency restoration
    print("Final frequency:", df.index.freq)
    print("df length:", len(df))

    # mean of demand
    print("mean demand: ", df["demand"].mean())

    df["demand"] = df["demand"].clip(lower=0)  # Set negative values to zero
    print("mean demand: ", df["demand"].mean())
    df["pv"] = df["pv"].clip(lower=0)  # Set negative values to zero

    return df


df = get_data()

# Create an energy system
energy_system = solph.EnergySystem(timeindex=df.index)  # , infer_last_interval=False)

# Define an electricity bus
b_el = solph.buses.Bus(label="b_el")
energy_system.add(b_el)

# Define a "grid" bus representing the external grid
b_pv = solph.buses.Bus(label="pv_bus")
energy_system.add(b_pv)

# pv_link = solph.components.Link(
#     label="pv_link",
#     inputs={b_pv: solph.flows.Flow()},
#     outputs={b_el: solph.flows.Flow()},
#     conversion_factors={(b_pv, b_el): 1},
# )
pv_consumption_compensation = 28.74  # €cent/kWh
pv_link = solph.components.Converter(
    label="pv_link",
    inputs={b_pv: solph.flows.Flow()},
    outputs={b_el: solph.flows.Flow(min=0, nominal_value=1000, fixed_costs=pv_consumption_compensation)},  # Prevents flow back into b_pv
    conversion_factors={(b_pv, b_el): 1},
)

energy_system.add(pv_link)

# PV system as a source
pv = solph.components.Source(
    label="pv_source", outputs={b_pv: solph.flows.Flow(fix=df["pv"], nominal_value=1)}
)
energy_system.add(pv)

# pv feed-in sink (selling electricity at a constant price)
excess_sink = solph.components.Sink(
    label="excess_sink", inputs={b_pv: solph.flows.Flow(fixed_costs=feed_in_price)}
)
energy_system.add(excess_sink)


# Grid source with variable electricity prices
el_source = solph.components.Source(
    label="el_source", outputs={b_el: solph.flows.Flow(variable_costs=df["price"])}
)
energy_system.add(el_source)

# Electricity demand as a sink
demand = solph.components.Sink(
    label="demand_sink",
    inputs={b_el: solph.flows.Flow(fix=df["demand"], nominal_value=1)},
)
energy_system.add(demand)


# Battery storage system
storage = solph.components.GenericStorage(
    label="storage",
    inputs={b_el: solph.flows.Flow(nominal_value=100)},  # 100 kW charge power
    outputs={b_el: solph.flows.Flow(nominal_value=100)},  # 100 kW discharge power
    nominal_storage_capacity=400,  # 400 kWh total storage capacity
    initial_storage_level=0.5,
    loss_rate=0.001,
    balanced=True,
    inflow_conversion_factor=0.95,
    outflow_conversion_factor=0.95,
)
energy_system.add(storage)

# Create and solve the optimization model
model = solph.Model(energy_system)

model.solve(solver="cbc", solve_kwargs={"tee": True})

# Extract results
results = solph.processing.results(model)
print("Results keys:")
for k in results.keys():
    print(k)  # See what keys exist

meta_results = solph.processing.meta_results(model)

# Convert results to DataFrame for analysis
# storage_flows = solph.processing.results(model)["storage"]["sequences"]
storage_flows = results[(storage, None)]["sequences"]

# Save results
storage_flows.to_csv("storage_results.csv")

# Print key results
print(meta_results)
print(storage_flows.head())


# get storage content
def get_storage_content():
    # Get storage content (SOC)
    soc = results[(storage, None)]["sequences"]["storage_content"]

    # Compute battery charging and discharging power from SOC changes
    battery_power = soc.diff().shift(-1).fillna(0)  # Change in storage content per timestep
    return soc, battery_power
soc, battery_power = get_storage_content()

# Define start and end time for the plot
start_time = "2024-06-01"
end_time = "2024-07-30"

# Filter data to the selected time range
soc = soc.loc[start_time:end_time]
battery_power = battery_power.loc[start_time:end_time]
df = df.loc[start_time:end_time]


def plot_results():
    # PLOTTING
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # First subplot: demand, PV, Price, Battery Power
    ax1 = axes[0]
    demand_results = results[(b_el, demand)]["sequences"]["flow"].loc[start_time:end_time]
    pv_results = results[(pv, b_pv)]["sequences"]["flow"].loc[start_time:end_time]
    grid_import = results[(el_source, b_el)]["sequences"]["flow"].loc[
        start_time:end_time
    ]
    grid_export = results[(b_pv, excess_sink)]["sequences"]["flow"].loc[
        start_time:end_time
    ]

    ax1.plot(demand_results.index, demand_results, label="Demand", color="blue")
    ax1.plot(pv_results.index, pv_results, label="PV Production", color="orange")
    ax1.plot(battery_power.index, battery_power, label="Battery Power", color="red")
    ax1.plot(grid_import.index, grid_import, label="Grid Import", color="black")
    ax1.plot(grid_export.index, -grid_export, label="Grid Export", color="gray")

    ax1.set_ylabel("Power [kW]")
    ax1.legend(loc="upper left")
    # Add a single horizontal helper line at zero
    ax1.axhline(0, color="black", linestyle="dashed")

    # Add secondary axis for price
    ax2 = ax1.twinx()
    ax2.plot(
        df.index,
        df["price"],
        label="Electricity Price",
        color="black",
        linestyle="dotted",
    )
    ax2.set_ylabel("Price [€cent/kWh]")
    ax2.legend(loc="upper right")

    # --- Align Zero Levels ---
    def align_zero_levels(ax1, ax2):
        # Get limits
        power_min, power_max = ax1.get_ylim()
        price_min, price_max = ax2.get_ylim()
        
        if power_min == 0 or price_min == 0:
            return  # Avoid division by zero issues

        power_ratio = power_max / power_min
        price_ratio = price_max / price_min

        if power_ratio < price_ratio:
            price_max = price_min * power_ratio
            print("adjusting price_max: ", price_max)
        else:
            price_min = price_max / power_ratio
            print("adjusting price_min: ", price_min)

        ax1.set_ylim(power_min, power_max)
        ax2.set_ylim(price_min, price_max)

    align_zero_levels(ax1, ax2)

    ax1.set_title(
        "Energy demand, PV Production, Battery Power, and Electricity Price"
    )

    # Second subplot: State of Charge (SOC)
    ax3 = axes[1]
    ax3.plot(soc.index, soc, label="Battery SOC", color="purple")
    ax3.set_ylabel("State of Charge [kWh]")
    ax3.axhline(0, color="gray", linestyle="dashed")  # Horizontal helper line for SOC
    # ax3.spines["left"].set_position(("data", 0))  # Move y-axis to cross x-axis at zero
    ax3.legend()
    ax3.set_title("Battery State of Charge")

    # Align y-axis zero levels
    ax1.set_ylim(min(ax1.get_ylim()[0], 0), ax1.get_ylim()[1])
    ax2.set_ylim(min(ax2.get_ylim()[0], 0), ax2.get_ylim()[1])
    ax3.set_ylim(min(ax3.get_ylim()[0], 0), ax3.get_ylim()[1])

    # Final formatting
    axes[1].set_xlabel("Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_results()
