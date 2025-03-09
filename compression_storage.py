import pandas as pd
import oemof.solph as solph
import matplotlib.pyplot as plt
import sys
from pyomo.environ import Var, ConstraintList, Binary
from pyomo.opt import SolverFactory
from tabulate import tabulate

# Set to True to enable non-simultaneity constraints
switch_non_simultaneity = False


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

    df["demand"] = df["demand"].clip(lower=0)  # Set negative values to zero
    df["pv"] = df["pv"].clip(lower=0)  # Set negative values to zero

    return df


df = get_data()

# Define a constant feed-in price in €/Wh
feed_in_price = 14  # €cent/kWh
pv_consumption_compensation = 0 # 28.74  # €cent/kWh
factor = 0.4
heat_price = 7  # €cent/kWh
cold_price = df["price"] - 4  # €cent/kWh
peak_threshold = 60  # kW
peak_cost = 20000  # €c/kW
converter_costs = 0.1  # €c/kWh 0.1 is quite high

# Create an energy system
energy_system = solph.EnergySystem(timeindex=df.index)

# Define an electricity bus
b_el = solph.buses.Bus(label="b_el")
b_pv = solph.buses.Bus(label="pv_bus")
b_air = solph.buses.Bus(label="b_air")  # Air Bus
b_heat = solph.buses.Bus(label="b_heat")  # Heat Output Bus
b_cold = solph.buses.Bus(label="b_cold")  # Cold Output Bus
energy_system.add(b_el, b_pv, b_air, b_air, b_heat, b_cold)


pv_link = solph.components.Converter(
    label="pv_link",
    inputs={b_pv: solph.flows.Flow()},
    outputs={
        b_el: solph.flows.Flow(
            min=0, nominal_value=1000, variable_costs=-pv_consumption_compensation
        )
    },  # Prevents flow back into b_pv
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
    label="excess_sink", inputs={b_pv: solph.flows.Flow(variable_costs=-feed_in_price)}
)
energy_system.add(excess_sink)

# Grid source with variable electricity prices
el_source = solph.components.Source(
    label="el_source", outputs={b_el: solph.flows.Flow(nominal_value=peak_threshold, variable_costs=df["price"]*1.2)}
)
energy_system.add(el_source)

# Peak electricity source with higher variable costs
el_peak_source = solph.components.Source(
    label="el_peak_source",
    outputs={b_el: solph.flows.Flow(variable_costs=df["price"] + peak_cost)},
)
energy_system.add(el_peak_source)

# Electricity demand as a sink
demand = solph.components.Sink(
    label="demand_sink",
    inputs={b_el: solph.flows.Flow(fix=df["demand"], nominal_value=1)},
)
energy_system.add(demand)

# storage system
storage = solph.components.GenericStorage(
    label="storage",
    inputs={b_air: solph.flows.Flow(nominal_value=100)},  # 100 kW charge power
    outputs={b_air: solph.flows.Flow(nominal_value=100)},  # 100 kW discharge power
    nominal_storage_capacity=400,  # 400 kWh total storage capacity
    initial_storage_level=0.5,
    loss_rate=0,  # No self-discharge expected
    balanced=True,
    inflow_conversion_factor=1,
    outflow_conversion_factor=1,
)
energy_system.add(storage)

# Compression Process: Electricity → Compressed Air + Heat
compression_converter = solph.components.Converter(
    label="compression_converter",
    inputs={b_el: solph.flows.Flow(nominal_value=100, variable_costs=converter_costs)},  # Max input power 100 kW
    outputs={
        b_air: solph.flows.Flow(nominal_value=100),  # Storing compressed air
        b_heat: solph.flows.Flow(nominal_value=90),  # Extracting heat
    },
    conversion_factors={
        b_air: 1,  # 100 kWh of electricity goes into 100 kWh compressed air
        b_heat: 0.9,  # 90 kWh heat extracted during compression
    }
)
energy_system.add(compression_converter)

# Expansion Process: Compressed Air → Electricity + Cold
expansion_converter = solph.components.Converter(
    label="expansion_converter",
    inputs={
        b_air: solph.flows.Flow(nominal_value=100, variable_costs=converter_costs)
    },  # Max air input 100 kW
    outputs={
        b_el: solph.flows.Flow(nominal_value=40),  # 40 kWh recovered as electricity
        b_cold: solph.flows.Flow(nominal_value=40),  # 40 kWh cold output
    },
    conversion_factors={
        b_el: 0.4,  # 40% of stored air energy converted to electricity
        b_cold: 0.4,  # 40% of stored air energy converted to cold
    },
)
energy_system.add(expansion_converter)

# sink for heat
heat_sink = solph.components.Sink(
    label="heat_sink", inputs={b_heat: solph.flows.Flow(variable_costs=-heat_price)}
)
energy_system.add(heat_sink)

cold_sink = solph.components.Sink(
    label="cold_sink", inputs={b_cold: solph.flows.Flow(variable_costs=-cold_price)}
)
energy_system.add(cold_sink)

# Create and solve the optimization model
model = solph.Model(energy_system)

def apply_non_simultaneity_constraints(model, compression_converter, expansion_converter, b_air, enable=True):
    if not enable:
        return

    ## apply big-M method to enforce non-simultaneity of compression and expansion
    # --- Add binary variables ---
    # non_simul_mode[t] == 1 means that in time period t the compression converter is allowed to run,
    # and the expansion converter is forced off; if 0 then the reverse holds.

    # Note: Replace the indices below with the correct keys to access the flow variables
    # from the respective converter components in your pyomo model.
    time_keys = sorted(
        {
            key[2]
            for key in model.flow.keys()
            if key[0] == compression_converter and key[1] == b_air
        }
    )
    model.non_simul_mode = Var(time_keys, domain=Binary)

    # --- Add constraints to enforce non-simultaneity ---
    model.non_simul_constraints = ConstraintList()

    M = 100

    for t in time_keys:
        # Constraint: If non_simul_mode[t] is 1, compression flow can be up to M, but expansion must be 0.
        model.non_simul_constraints.add(
            model.flow[compression_converter, b_air, t]
            <= M * model.non_simul_mode[t]
        )
        # Constraint: If non_simul_mode[t] is 0, expansion flow can be up to M, but compression must be 0.
        model.non_simul_constraints.add(
            model.flow[b_air, expansion_converter, t]
            <= M * (1 - model.non_simul_mode[t])
        )

# Apply the non-simultaneity constraints
apply_non_simultaneity_constraints(model, compression_converter, expansion_converter, b_air, enable=switch_non_simultaneity)


# solver = SolverFactory("cbc")
# solver.options["threads"] = 8
# Relax the optimality gap to 5% or 10%
# solver.options["ratioGap"] = 0.1  # or 5%, meaning a 5% gap is acceptable

# results = solver.solve(model, tee=True)
# model.solutions.load_from(results)
# print("Solver Status:", results.solver.status)
# print("Termination Condition:", results.solver.termination_condition)

# model.solve(solver="cbc", solve_kwargs={"tee": True})
model.solve(solver="cbc", solve_kwargs={"tee": True, "options": {"ratioGap": 0.1}})

# Extract results
results = solph.processing.results(model)

# print("Results keys:")
# for k in results.keys():
#     print(k)  # See what keys exist

meta_results = solph.processing.meta_results(model)

# Convert results to DataFrame for analysis
# storage_flows = solph.processing.results(model)["storage"]["sequences"]
storage_flows = results[(storage, None)]["sequences"]

# Save results
storage_flows.to_csv("storage_results.csv")

# Print key results
print(meta_results)
print(storage_flows.head())

# Define start and end time for the plot
start_time = "2024-05-01"
end_time = "2024-05-16"

def preprocess_ref(df):
    df["grid_import_ref"] = (df["demand"] - df["pv"]).clip(lower=0)
    df["pv_feed_in_ref"] = (df["pv"] - df["demand"]).clip(lower=0)
    df["pv_self_use_ref"] = df["pv"] - df["pv_feed_in_ref"]
    return df

def preprocess_caes(df, results):
    # Transfer all relevant caes result flows into df columns.
    df["grid_import_caes"] = (
        results[(el_source, b_el)]["sequences"]["flow"]
        + results[(el_peak_source, b_el)]["sequences"]["flow"]
    ).loc[df.index]
    df["pv_feed_in_caes"] = results[(b_pv, excess_sink)]["sequences"]["flow"].loc[df.index]
    df["pv_self_use_caes"] = df["pv"] - df["pv_feed_in_caes"]
    df["compression_power"] = results[(b_el, compression_converter)]["sequences"]["flow"].loc[df.index]
    df["expansion_power"] = results[(expansion_converter, b_el)]["sequences"]["flow"].loc[df.index]
    df["heat_output"] = results[(compression_converter, b_heat)]["sequences"]["flow"].loc[df.index]
    df["cold_output"] = results[(expansion_converter, b_cold)]["sequences"]["flow"].loc[df.index]
    df["soc"] = results[(storage, None)]["sequences"]["storage_content"].loc[df.index]
    return df

def save_preprocessed_df(df):
    # filename = input("Enter the filename to save the preprocessed dataframe: ")
    # if filename:
    #     filename = "results/" + filename + ".csv"
    # else:
    filename = "results/results.csv"
    df.to_csv(filename)


def recalculate_compression_expansion(df, results):
    """
    Recalculates compression and expansion power based on SOC to avoid simultaneous compression and expansion.
    Also updates heat output, cold output, and adjusts grid import, PV self-use and PV feed-in accordingly.
    """
    df["soc"] = results[(storage, None)]["sequences"]["storage_content"].loc[df.index]

    # Compute storage power change per timestep
    soc_change = (
        df["soc"].diff().shift(-1).fillna(0)
    )  # Forward shift to align with time step changes

    # Create new series for recalculated values
    df["compression_power"] = soc_change.clip(
        lower=0
    )  # Only positive changes = compression
    expansion_efficiency = 0.4
    df["expansion_power"] = -soc_change.clip(
        upper=0
    )  * expansion_efficiency  # Only negative changes = expansion

    # Compute heat and cold outputs
    heat_output_efficiency = 0.9  # heat recovery from compression
    cold_output_efficiency = 0.4  # cooling output from expansion

    df["heat_output"] = df["compression_power"] * heat_output_efficiency
    df["cold_output"] = df["expansion_power"] * cold_output_efficiency

    energy_balance = (
        df["pv"] + df["expansion_power"] - df["compression_power"] - df["demand"]
    )

    df["pv_feed_in_caes"] = energy_balance.clip(lower=0)
    df["pv_self_use_caes"] = df["pv"] - df["pv_feed_in_caes"]
    df["grid_import_caes"] = (-energy_balance).clip(lower=0)

    excess_energy = (
        df["grid_import_caes"]
        + df["pv_feed_in_caes"]
        + df["expansion_power"]
        - df["compression_power"]
        - df["demand"]
    )

    # Check for significant mismatch
    tolerance = 1e-3
    if abs(excess_energy.sum()) > tolerance:
        print("⚠️ Warning: Energy balance mismatch detected!")

    print("\n✅ Recalculated compression & expansion power based on SOC")
    print("   ➝ Sum of excess energy:", excess_energy.sum())
    print("   ➝ No simultaneous bidirectional flows")
    print("   ➝ Adjusted grid import & PV self-use accordingly")
    return df


def validate_caes_model(df, results):
    print("\n🔍 Running CAES Model Validation Checks...")

    try:
        # 1️⃣ Demand and PV production should remain identical
        assert (
            df["demand"].sum() == results[(b_el, demand)]["sequences"]["flow"].sum()
        ), "Mismatch in demand"
        assert (
            df["pv"].sum() == results[(pv, b_pv)]["sequences"]["flow"].sum()
        ), "Mismatch in PV production"

        # 2️⃣ Total electricity consumption should increase
        total_energy_ref = (df["demand"] - df["pv"]).clip(lower=0).sum()
        total_energy_caes = (
            df["grid_import_caes"].sum()
            + df["pv_self_use_caes"].sum()
            - df["compression_power"].sum()
            + df["expansion_power"].sum()
        )
        print("total_energy_ref: ", total_energy_ref)
        print("total_energy_caes: ", total_energy_caes)
        assert (
            total_energy_caes >= total_energy_ref
        ), "⚠️ Total energy consumption should increase with CAES"

        # 3️⃣ PV link flow is always positive
        assert df["pv_self_use_caes"].min() >= 0, "⚠️ PV link flow should always be positive"

        # 4️⃣ Only PV should feed the excess sink
        pv_production = results[(pv, b_pv)]["sequences"]["flow"]
        excess_sink_series = results[(b_pv, excess_sink)]["sequences"]["flow"]
        assert (
            pv_production - excess_sink_series
        ).min() >= 0, "⚠️ Only PV should feed the excess sink"

        # 5️⃣ Stored compressed air should match expanded air
        b_air_in_series = results[(compression_converter, b_air)]["sequences"][
            "flow"
        ]
        b_air_out_series = results[(b_air, expansion_converter)]["sequences"][
            "flow"
        ]
        assert (
            b_air_in_series.sum() - b_air_out_series.sum() < 1e-3
        ), "⚠️ Stored air does not match expanded air"

        # 6️⃣ Expansion-to-Compression efficiency should be 40%
        compression_energy = df["compression_power"].sum()
        expansion_energy = df["expansion_power"].sum()
        assert (
            abs(compression_energy - expansion_energy / 0.4) < 1e-3
        ), "⚠️ CAES electric efficiency is incorrect"

        # 7️⃣ Heat & Cold outputs should match expected ratios
        heat_output = df["heat_output"].sum()
        cold_output = df["cold_output"].sum()
        print(f"cold output ratio: {cold_output / expansion_energy}")
        assert (
            abs(heat_output / compression_energy - 0.9) < 1e-3
        ), "⚠️ Heat output ratio incorrect"
        # Cold output should be 40% of b_air_out_series
        assert (
            abs(cold_output / b_air_out_series.sum() - 0.4) < 1e-3
                ), "⚠️ Cold output ratio incorrect"
        assert (
            abs(cold_output - expansion_energy) < 1e-3
        ), "⚠️ Cold output ratio incorrect"

        # 8️⃣ Heat and cold output ratios to stored air
        assert (
            abs(heat_output / b_air_in_series.sum() - 0.9) < 1e-3
        ), "⚠️ Heat output ratio incorrect"
        assert (
            abs(cold_output / b_air_out_series.sum() - 0.4) < 1e-3
        ), "⚠️ Cold output ratio incorrect"

        # 9️⃣ PV self-consumption plus feed-in should match PV production
        assert (
            abs(df["pv_self_use_caes"].sum() + df["pv_feed_in_caes"].sum() - df["pv"].sum()) < 1e-3
        ), "⚠️ PV self-consumption and feed-in mismatch"

        #  🔟 **Balance Calculation: Energy Inputs vs. Outputs**
        print("\n📊 **Energy Balance Check** 📊")
        print("-" * 50)

        # Now using the recalculated values from df:
        total_energy_in = (
            df["grid_import_caes"].sum() + df["pv"].sum() + expansion_energy
        )
        total_energy_out = (
            compression_energy + df["demand"].sum() + df["pv_feed_in_caes"].sum()
        )

        print(f'🔹 Grid Import: {df["grid_import_caes"].sum():.2f} kWh')
        print(f'🔹 PV Generation: {df["pv"].sum():.2f} kWh')
        print(f"🔹 Expansion Power: {expansion_energy:.2f} kWh")
        print(f"-----------------------------------")
        print(f"🔸 Compression Power: {compression_energy:.2f} kWh")
        print(f'🔸 Demand: {df["demand"].sum():.2f} kWh')
        print(f'🔸 PV Feed-in: {df["pv_feed_in_caes"].sum():.2f} kWh')
        print(f"-----------------------------------")
        print(f"✅ Total Energy In: {total_energy_in:.2f} kWh")
        print(f"✅ Total Energy Out: {total_energy_out:.2f} kWh")
        print(f"-----------------------------------")

        assert (
            abs(total_energy_in - total_energy_out) < 1e-3
        ), "⚠️ Energy balance mismatch"

        print("✅ All validation checks passed successfully!")

    except AssertionError as e:
        print(f"❌ Validation failed: {e}")


def evaluate_economic_impact(df, results):
    print("\n📊 Evaluating Economic Impact of CAES...")
    # Compute grid import (if demand exceeds PV)
    print("grid_import_ref: ", df["grid_import_ref"].sum())
    print("pv_ref total: ", df["pv"].sum())
    print("demand_ref total: ", df["demand"].sum())

    print("pv_feed_in_ref: ", df["pv_feed_in_ref"].sum())

    # Compute PV self-use as the remaining PV after accounting for feed-in
    print("pv_self_use_ref: ", df["pv_self_use_ref"].sum())

    # Earnings from using PV directly (self-consumption)
    df["pv_self_use_earnings_ref"] = df["pv_self_use_ref"] * (
        pv_consumption_compensation / 100
    )  # Convert to €
    # Earnings from exporting excess PV
    df["pv_feed_in_earnings_ref"] = df["pv_feed_in_ref"] * (
        feed_in_price / 100
    )  # Convert to €

    # Grid import cost
    df["cost_grid_import_ref"] = df["grid_import_ref"] * (
        df["price"] / 100
    )  # Convert to €
    # peak cost reference
    peak_cost_ref = df["grid_import_ref"].max() * peak_cost / 100

    # Total cost in the reference case
    df["total_cost_ref"] = (
        df["cost_grid_import_ref"]
        - df["pv_self_use_earnings_ref"]
        - df["pv_feed_in_earnings_ref"]
    )

    # CAES case (optimized system)
    print("grid_import_caes: ", df["grid_import_caes"].sum())
    print("compression_power: ", df["compression_power"].sum())
    print("expansion_power: ", df["expansion_power"].sum())
    # print("^difference: ", df["compression_power"].sum() - df["expansion_power"].sum())
    print("heat_output: ", df["heat_output"].sum())
    print("cold_output: ", df["cold_output"].sum())
    print("pv_self_use_caes: ", df["pv_self_use_caes"].sum())

    df["cost_grid_import_caes"] = (
        df["grid_import_caes"] * df["price"] / 100
    )  # Convert to €
    cost_peak_caes = df["grid_import_caes"].max() * peak_cost / 100
    df["pv_self_use_earnings_caes"] = (
        df["pv_self_use_caes"] * pv_consumption_compensation / 100
    )  # Convert to €
    df["pv_feed_in_earnings_caes"] = (
        df["pv_feed_in_caes"] * feed_in_price / 100
    )  # Convert to €

    df["heat_earnings_caes"] = df["heat_output"] * heat_price / 100
    df["cold_earnings_caes"] = df["cold_output"] * (df["price"] - 5) / 100

    df["total_cost_caes"] = (
        df["cost_grid_import_caes"]
        - df["pv_self_use_earnings_caes"]
        - df["pv_feed_in_earnings_caes"]
        - df["heat_earnings_caes"]
        - df["cold_earnings_caes"]
    )

    # Economic impact of CAES
    df["cost_savings"] = df["total_cost_ref"] - df["total_cost_caes"]
    df["grid_import_reduction"] = df["grid_import_ref"] - df["grid_import_caes"]
    df["pv_self_use_increase"] = df["pv_self_use_caes"] - df["pv_self_use_ref"]
    df["pv_feed_in_difference"] = (
        df["pv_feed_in_earnings_caes"] - df["pv_feed_in_earnings_ref"]
    )
    peak_cost_difference = peak_cost_ref - cost_peak_caes
    cost_savings = df["cost_savings"].sum() + peak_cost_difference

    print("\n### Economic Impact of CAES ###")
    print(f"Total cost savings: {cost_savings:.2f} €")
    print(f"Thereof peak cost difference: {peak_cost_difference:.2f} €")
    print(f"Grid import reduction: {df['grid_import_reduction'].sum():.2f} kWh")
    print(f"PV self-consumption increase: {df['pv_self_use_increase'].sum():.2f} kWh")
    print(f"PV feed-in revenue change: {df['pv_feed_in_difference'].sum():.2f} €")
    print(f"Heat earnings: {df['heat_earnings_caes'].sum():.2f} €")
    print(f"Cold earnings: {df['cold_earnings_caes'].sum():.2f} €")

    # Warning if CAES increases cost instead of reducing it
    if df["cost_savings"].sum() < 0:
        print(
            "\n⚠️ Warning: CAES increased total costs instead of reducing them! Review input assumptions."
        )


def plot_results(start_time, end_time, df):
    # PLOTTING
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # First subplot: demand, PV, Price, Battery Power
    ax1 = axes[0]
    grid_import = df["grid_import_caes"]
    max_grid_import = grid_import.max()
    print("max grid import: ", max_grid_import)

    if max_grid_import > peak_threshold:
        peak_time = grid_import.idxmax()
        # print date and time of max grid import
        print("#########################")
        print("max grid import > peak_threshold")
        print("max grid import date and time: ", peak_time)
        print("#########################")

        # adjust start_time and end_time to show the peak time
        # calculate delta based on start_time and end_time
        delta = pd.to_datetime(end_time) - pd.to_datetime(start_time)
        start_time = peak_time - delta / 2
        end_time = peak_time + delta / 2
        print("plot adjusted to show peak time")
        print("start_time: ", start_time)
        print("end_time: ", end_time)

        df_cut = df.loc[start_time:end_time]

        ax1.axhline(max_grid_import, color="black", linestyle="-")
        # Add text annotation above the line
        ax1.text(
            x=df_cut.index[
                int(len(df_cut.index) * 0.5)
            ],  # Place it at ~80% of the x-axis
            y=max_grid_import + 3,  # Slightly above the line
            s="Real max peak - threshold broken!",
            color="red",
            fontsize=10,
            fontweight="bold",
            ha="center",  # Center align text
        )
    else:
        df_cut = df.loc[start_time:end_time]

    demand_results = df_cut["demand"]
    pv_results = df_cut["pv"]
    grid_export = df_cut["pv_feed_in_caes"]
    compression_power = df_cut["compression_power"]
    expansion_power = df_cut["expansion_power"]

    ax1.axhline(peak_threshold, color="red", linestyle="dashed")
    ax1.plot(demand_results.index, demand_results, label="Demand", color="blue")
    ax1.plot(pv_results.index, pv_results, label="PV Production", color="orange")

    grid_import_cut = grid_import.loc[start_time:end_time]
    ax1.plot(grid_import_cut.index, grid_import_cut, label="Grid Import", color="black")

    ax1.plot(grid_export.index, -grid_export, label="Grid Export", color="gray")
    ax1.plot(
        compression_power.index,
        -compression_power,
        label="Compression Power",
        color="purple",
    )
    ax1.plot(
        expansion_power.index, expansion_power, label="Expansion Power", color="green"
    )

    # pv_link_flow = results[(pv_link, b_el)]["sequences"]["flow"]
    # ax1.plot(pv_link_flow.index, pv_link_flow, label="PV Link Flow", color="brown")

    ax1.set_ylabel("Power [kW]")
    ax1.legend(loc="lower left")
    # Add a single horizontal helper line at zero
    ax1.axhline(0, color="black", linestyle="dashed")

    # Add secondary axis for price
    ax2 = ax1.twinx()
    # df_cut = df.loc[start_time:end_time]
    ax2.plot(
        df_cut.index,
        df_cut["price"],
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
        else:
            price_min = price_max / power_ratio

        ax1.set_ylim(power_min, power_max)
        ax2.set_ylim(price_min, price_max)

    align_zero_levels(ax1, ax2)

    ax1.set_title("Energy demand, PV Production, Battery Power, and Electricity Price")

    # Second subplot: State of Charge (SOC)
    ax3 = axes[1]
    ax3.plot(df_cut.index, df_cut["soc"], label="Battery SOC", color="purple")
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


def plot_cost_series(df):
    """
    Plot cost series for the reference and CAES case in a single figure with consistent colors.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 🎨 **Define Color Mapping**
    colors = {
        "total_cost": "black",
        "grid_import": "gray",
        "pv_feed_in": "orange",
        "pv_self_use": "yellow",
        "heat_earnings": "red",
        "cold_earnings": "blue",
        "compression_cost": "purple",
        "expansion_cost": "green",
    }

    # 📊 **Plot Reference Costs**
    ax = axes[0]
    ax.plot(
        df.index,
        df["cost_grid_import_ref"].cumsum(),
        label="Grid Import Costs (Reference)",
        color=colors["grid_import"],
    )
    ax.plot(
        df.index,
        -df["pv_feed_in_earnings_ref"].cumsum(),
        label="PV Feed-In Earnings (Reference)",
        color=colors["pv_feed_in"],
    )
    ax.plot(
        df.index,
        -df["pv_self_use_earnings_ref"].cumsum(),
        label="PV Self-Use Earnings (Reference)",
        color=colors["pv_self_use"],
    )
    ax.plot(
        df.index,
        df["total_cost_ref"].cumsum(),
        label="Total Costs (Reference)",
        color=colors["total_cost"],
        linewidth=1.5,
    )

    ax.set_ylabel("Costs [€]")
    ax.set_title("Reference Case Costs")
    ax.legend()

    # 📊 **Plot CAES Costs**
    ax = axes[1]
    ax.plot(
        df.index,
        df["cost_grid_import_caes"].cumsum(),
        label="Grid Import Costs (CAES)",
        color=colors["grid_import"],
    )
    ax.plot(
        df.index,
        -df["pv_feed_in_earnings_caes"].cumsum(),
        label="PV Feed-In Earnings (CAES)",
        color=colors["pv_feed_in"],
    )
    ax.plot(
        df.index,
        -df["pv_self_use_earnings_caes"].cumsum(),
        label="PV Self-Use Earnings (CAES)",
        color=colors["pv_self_use"],
    )
    ax.plot(
        df.index,
        -df["heat_earnings_caes"].cumsum(),
        label="Heat Earnings (CAES)",
        color=colors["heat_earnings"],
    )
    ax.plot(
        df.index,
        -df["cold_earnings_caes"].cumsum(),
        label="Cold Earnings (CAES)",
        color=colors["cold_earnings"],
    )
    ax.plot(
        df.index,
        (df["compression_power"] * converter_costs / 100).cumsum(),
        label="Compression Costs (CAES)",
        color=colors["compression_cost"],
    )
    ax.plot(
        df.index,
        (df["expansion_power"] * converter_costs / 100).cumsum(),
        label="Expansion Costs (CAES)",
        color=colors["expansion_cost"],
    )
    ax.plot(
        df.index,
        df["total_cost_caes"].cumsum(),
        label="Total Costs (Net) (CAES)",
        color=colors["total_cost"],
        linewidth=1.5,
    )

    ax.set_ylabel("Costs [€]")
    ax.set_xlabel("Time")
    ax.set_title("CAES Case Costs")
    ax.legend()

    # Formatting
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def create_energy_balance_table(df):
    total_energy_in_ref = df["grid_import_ref"].sum() + df["pv"].sum()
    total_energy_out_ref = df["demand"].sum() + df["pv_feed_in_ref"].sum()

    total_energy_in_caes = (
        df["grid_import_caes"].sum() + df["pv"].sum() + df["expansion_power"].sum()
    )
    total_energy_out_caes = (
        df["demand"].sum() + df["pv_feed_in_caes"].sum() + df["compression_power"].sum()
    )

    balance_data = {
        "Reference [kWh]": [
            df["demand"].sum(),
            df["pv"].sum(),
            df["pv_feed_in_ref"].sum(),
            df["grid_import_ref"].sum(),
            None,
            None,
            total_energy_in_ref,
            total_energy_out_ref,
            total_energy_in_ref - total_energy_out_ref,
        ],
        "CAES [kWh]": [
            df["demand"].sum(),
            df["pv"].sum(),
            df["pv_feed_in_caes"].sum(),
            df["grid_import_caes"].sum(),
            df["compression_power"].sum(),
            df["expansion_power"].sum(),
            total_energy_in_caes,
            total_energy_out_caes,
            total_energy_in_caes - total_energy_out_caes,
        ],
    }

    index = [
        "Total Demand",
        "PV Generation",
        "PV Feed-In",
        "Grid Import",
        "Compression Energy",
        "Expansion Energy",
        "Total Energy In",
        "Total Energy Out",
        "Energy Balance (In - Out)",
    ]

    df_energy = pd.DataFrame(balance_data, index=index)
    df_energy.to_csv("results/energy_balance.csv")
    print(tabulate(df_energy, headers="keys", tablefmt="fancy_grid", floatfmt=".2f"))


def create_economic_summary_table(df):
    peak_cost_ref = df["grid_import_ref"].max() * peak_cost / 100
    cost_peak_caes = df["grid_import_caes"].max() * peak_cost / 100

    economics_data = {
        "Reference [€]": [
            df["cost_grid_import_ref"].sum(),
            peak_cost_ref,
            -df["pv_feed_in_earnings_ref"].sum(),
            -df["pv_self_use_earnings_ref"].sum(),
            0,
            0,
            0,
            0,
            df["total_cost_ref"].sum(),
        ],
        "CAES [€]": [
            df["cost_grid_import_caes"].sum(),
            cost_peak_caes,
            -df["pv_feed_in_earnings_caes"].sum(),
            -df["pv_self_use_earnings_caes"].sum(),
            -df["heat_earnings_caes"].sum(),
            -df["cold_earnings_caes"].sum(),
            df["compression_power"].sum() * converter_costs / 100,
            df["expansion_power"].sum() * converter_costs / 100,
            df["total_cost_caes"].sum(),
        ],
    }

    economics_df = pd.DataFrame(
        economics_data,
        index=[
            "Grid Import Costs",
            "Peak Load Costs",
            "PV Feed-in Earnings",
            "PV Self-Consumption Earnings",
            "Heat Earnings",
            "Cold Earnings",
            "Compression Costs",
            "Expansion Costs",
            "Total Costs (Net)",
        ],
    )

    economics_df["Difference [€]"] = (
        economics_df["Reference [€]"] - economics_df["CAES [€]"]
    )
    economics_df.to_csv("results/economic_summary.csv")
    print(tabulate(economics_df, headers="keys", tablefmt="fancy_grid", floatfmt=".2f"))


dfs_to_evaluate = []

# Preprocess results without recalculation
df = preprocess_ref(df)
df_original = preprocess_caes(df.copy(), results)
dfs_to_evaluate.append(("Optimization Results", df_original))

# Add recalculated results if recalculation is switched off (LP case)
if not switch_non_simultaneity:
    df_recalculated = recalculate_compression_expansion(df.copy(), results)
    dfs_to_evaluate.append(("Recalculated (SOC-based)", df_recalculated))

# Process and visualize all prepared DataFrames
for label, df_l in dfs_to_evaluate:
    print(f"\n🔹 Evaluating: {label}")
    save_preprocessed_df(df_l)
    validate_caes_model(df_l, results)
    evaluate_economic_impact(df_l, results)
    create_energy_balance_table(df_l)
    create_economic_summary_table(df_l)
    plot_results(start_time, end_time, df_l)
    plot_cost_series(df_l)
