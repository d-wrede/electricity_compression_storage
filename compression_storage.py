import pandas as pd
import oemof.solph as solph
import matplotlib.pyplot as plt


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
pv_consumption_compensation = 28.74  # €cent/kWh
factor = 0.4
heat_price = 7 * 0.2  # €cent/kWh
cold_price = df["price"] - 5  # €cent/kWh
peak_threshold = 60  # kW
peak_cost = 20000  # €c/kW
converter_costs = 0.1  # €c/kWh 0.1 is quite high

# Create an energy system
energy_system = solph.EnergySystem(timeindex=df.index)

# Define an electricity bus
b_el = solph.buses.Bus(label="b_el")
b_pv = solph.buses.Bus(label="pv_bus")
b_air_in = solph.buses.Bus(label="b_air_in")  # Air Input Bus
b_air_out = solph.buses.Bus(label="b_air_out")  # Air Output Bus
b_heat = solph.buses.Bus(label="b_heat")  # Heat Output Bus
b_cold = solph.buses.Bus(label="b_cold")  # Cold Output Bus
energy_system.add(b_el, b_pv, b_air_in, b_air_out, b_heat, b_cold)


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
    inputs={b_air_in: solph.flows.Flow(nominal_value=100)},  # 100 kW charge power
    outputs={b_air_out: solph.flows.Flow(nominal_value=100)},  # 100 kW discharge power
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
        b_air_in: solph.flows.Flow(nominal_value=100),  # Storing compressed air
        b_heat: solph.flows.Flow(nominal_value=90),  # Extracting heat
    },
    conversion_factors={
        b_air_in: 1,  # 100 kWh of electricity goes into 100 kWh compressed air
        b_heat: 0.9,  # 90 kWh heat extracted during compression
    }
)
energy_system.add(compression_converter)

# Expansion Process: Compressed Air → Electricity + Cold
expansion_converter = solph.components.Converter(
    label="expansion_converter",
    inputs={
        b_air_out: solph.flows.Flow(nominal_value=100, variable_costs=converter_costs)
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
start_time = "2024-01-01"
end_time = "2024-12-30"


# df = df.loc[start_time:end_time]
def preprocess_df(df, results):
    # Transfer all relevant result flows into df columns.
    df["grid_import"] = (
        results[(el_source, b_el)]["sequences"]["flow"]
        + results[(el_peak_source, b_el)]["sequences"]["flow"]
    ).loc[df.index]
    df["pv_feed_in"] = results[(b_pv, excess_sink)]["sequences"]["flow"].loc[df.index]
    df["pv_self_use"] = df["pv"] - df["pv_feed_in"]
    df["compression_power"] = results[(b_el, compression_converter)]["sequences"]["flow"].loc[df.index]
    df["expansion_power"] = results[(expansion_converter, b_el)]["sequences"]["flow"].loc[df.index]
    df["heat_output"] = results[(compression_converter, b_heat)]["sequences"]["flow"].loc[df.index]
    df["cold_output"] = results[(expansion_converter, b_cold)]["sequences"]["flow"].loc[df.index]

    return df


def recalculate_compression_expansion(df, results):
    """
    Recalculates compression and expansion power based on SOC to avoid simultaneous compression and expansion.
    Also updates heat output, cold output, and adjusts grid import, PV self-use and PV feed-in accordingly.
    """
    soc = results[(storage, None)]["sequences"]["storage_content"].loc[df.index]

    # Compute storage power change per timestep
    soc_change = (
        soc.diff().shift(-1).fillna(0)
    )  # Forward shift to align with time step changes

    # Create new series for recalculated values
    df["compression_power"] = soc_change.clip(
        lower=0
    )  # Only positive changes = compression
    df["expansion_power"] = -soc_change.clip(
        upper=0
    )  * 0.4  # Only negative changes = expansion

    # Compute heat and cold outputs
    df["heat_output"] = (
        df["compression_power"] * 0.9
    )  # 90% of compression energy
    df["cold_output"] = (
        df["expansion_power"] * 0.4
    )  # 40% of expansion energy

    # Get the original data from oemof results
    df["grid_import"] = results[(el_source, b_el)]["sequences"]["flow"].loc[df.index] + results[(el_peak_source, b_el)]["sequences"]["flow"].loc[df.index]
    df["pv_feed_in"] = results[(b_pv, excess_sink)]["sequences"]["flow"].loc[df.index]
    df["pv_self_use"] = df["pv"] - df["pv_feed_in"]  # PV used directly

    total_energy_in = (
        df["grid_import"] + df["pv_feed_in"] + df["expansion_power"]
    )
    total_energy_out = df["compression_power"] + df["demand"]

    df["excess_energy"] = total_energy_in - total_energy_out

    # known: df["pv"], df["demand"], df["price"]

    for t in df.index:
        # 1. check situation: is there excess energy?
        excess_energy = df.at[t, "excess_energy"]
        if excess_energy > 0:
            # 1️⃣ **Reduce Grid Import First**
            if excess_energy >= df.at[t, "grid_import"]:
                excess_energy -= df.at[t, "grid_import"]
                df.at[t, "grid_import"] = 0
            else:
                df.at[t, "grid_import"] -= excess_energy
                excess_energy = 0

            # 2️⃣ **If still excess energy, adjust PV self-use & feed-in**
            if excess_energy > 0:
                df.at[t, "pv_self_use"] -= excess_energy
                df.at[t, "pv_feed_in"] += excess_energy

        elif excess_energy < 0:
            # 3️⃣ **Increase PV Self-Use If Possible**
            if df.at[t, "pv_feed_in"] >= abs(excess_energy):
                df.at[t, "pv_feed_in"] -= abs(excess_energy)
                df.at[t, "pv_self_use"] += abs(excess_energy)
            else:
                # If PV is insufficient, increase grid import
                excess_energy += df.at[t, "pv_feed_in"]
                df.at[t, "pv_self_use"] += df.at[t, "pv_feed_in"]
                df.at[t, "pv_feed_in"] = 0
                df.at[t, "grid_import"] += abs(excess_energy)

    print("\n✅ Recalculated compression & expansion power based on SOC")
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
            df["grid_import"].sum()
            + df["pv_self_use"].sum()
            - df["compression_power"].sum()
            + df["expansion_power"].sum()
        )
        print("total_energy_ref: ", total_energy_ref)
        print("total_energy_caes: ", total_energy_caes)
        assert (
            total_energy_caes >= total_energy_ref
        ), "⚠️ Total energy consumption should increase with CAES"

        # 3️⃣ PV link flow is always positive
        assert df["pv_self_use"].min() >= 0, "⚠️ PV link flow should always be positive"

        # 4️⃣ Only PV should feed the excess sink
        pv_production = results[(pv, b_pv)]["sequences"]["flow"]
        excess_sink_series = results[(b_pv, excess_sink)]["sequences"]["flow"]
        assert (
            pv_production - excess_sink_series
        ).min() >= 0, "⚠️ Only PV should feed the excess sink"

        # 5️⃣ Stored compressed air should match expanded air
        b_air_in_series = results[(compression_converter, b_air_in)]["sequences"][
            "flow"
        ]
        b_air_out_series = results[(b_air_out, expansion_converter)]["sequences"][
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
        assert (
            abs(heat_output / compression_energy - 0.9) < 1e-3
        ), "⚠️ Heat output ratio incorrect"
        assert (
            abs(cold_output / expansion_energy - 0.4) < 1e-3
        ), "⚠️ Cold output ratio incorrect"

        # 8️⃣ Heat and cold output ratios to stored air
        assert (
            abs(heat_output / b_air_in_series.sum() - 0.9) < 1e-3
        ), "⚠️ Heat output ratio incorrect"
        assert (
            abs(cold_output / b_air_out_series.sum() - 0.4) < 1e-3
        ), "⚠️ Cold output ratio incorrect"

        #  🔟 **Balance Calculation: Energy Inputs vs. Outputs**
        print("\n📊 **Energy Balance Check** 📊")
        print("-" * 50)

        # Now using the recalculated values from df:
        total_energy_in = df["grid_import"].sum() + df["pv_feed_in"].sum() + expansion_energy
        total_energy_out = compression_energy + df["demand"].sum()

        print(f'🔹 Grid Import: {df["grid_import"].sum():.2f} kWh')
        print(f'🔹 PV Feed-in: {df["pv_feed_in"].sum():.2f} kWh')
        print(f'🔹 Expansion Power (recalc): {expansion_energy:.2f} kWh')
        print(f'-----------------------------------')
        print(f'🔸 Compression Power (recalc): {compression_energy:.2f} kWh')
        print(f'🔸 Demand: {df["demand"].sum():.2f} kWh')
        print(f'-----------------------------------')
        print(f'✅ Total Energy In: {total_energy_in:.2f} kWh')
        print(f'✅ Total Energy Out: {total_energy_out:.2f} kWh')
        print(f'-----------------------------------')

        assert (
            abs(total_energy_in - total_energy_out) < 1e-3
        ), "⚠️ Energy balance mismatch"

        print("✅ All validation checks passed successfully!")

    except AssertionError as e:
        print(f"❌ Validation failed: {e}")


def validate_caes_model_old(df, results):
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
            results[(el_source, b_el)]["sequences"]["flow"].sum()
            + results[(el_peak_source, b_el)]["sequences"]["flow"].sum()
            + results[(pv_link, b_el)]["sequences"]["flow"].sum()
        )
        print("total_energy_ref: ", total_energy_ref)
        print("total_energy_caes: ", total_energy_caes)
        assert (
            total_energy_caes >= total_energy_ref
        ), "⚠️ Total energy consumption should increase with CAES"

        # 3️⃣ pv link flow is always positive
        pv_link_flow = results[(pv_link, b_el)]["sequences"]["flow"]
        assert pv_link_flow.min() >= 0, "⚠️ PV link flow should always be positive"

        # 4️⃣ Only PV should feed the excess sink
        pv_production = results[(pv, b_pv)]["sequences"]["flow"]
        excess_sink_series = results[(b_pv, excess_sink)]["sequences"]["flow"]
        assert (
            pv_production - excess_sink_series
        ).min() >= 0, "⚠️ Only PV should feed the excess sink"

        # 5️⃣ Stored compressed air should match expanded air
        b_air_in_series = results[(compression_converter, b_air_in)]["sequences"][
            "flow"
        ]
        b_air_out_series = results[(b_air_out, expansion_converter)]["sequences"][
            "flow"
        ]
        assert (
            b_air_in_series.sum() - b_air_out_series.sum() < 1e-3
        ), "⚠️ Stored air does not match expanded air"

        # 6️⃣ Expansion-to-Compression efficiency should be 40%
        compression_power = results[(b_el, compression_converter)]["sequences"]["flow"]
        expansion_power = results[(expansion_converter, b_el)]["sequences"]["flow"]
        assert (
            abs(compression_power.sum() - expansion_power.sum() / 0.4) < 1e-3
        ), "⚠️ CAES electric efficiency is incorrect"

        # 7️⃣ Heat & Cold outputs should match expected ratios
        heat_output = results[(compression_converter, b_heat)]["sequences"]["flow"]
        cold_output = results[(expansion_converter, b_cold)]["sequences"]["flow"]
        assert (
            abs(heat_output.sum() / compression_power.sum() - 0.9) < 1e-3
        ), "⚠️ Heat output ratio incorrect"
        assert (
            abs(cold_output.sum() / expansion_power.sum() - 1.0) < 1e-3
        ), "⚠️ Cold output ratio incorrect"

        # 8️⃣ heat_ and cold_output ratio to b_air
        assert (
            abs(heat_output.sum() / b_air_in_series.sum() - 0.9) < 1e-3
        ), "⚠️ Heat output ratio incorrect"
        assert (
            abs(cold_output.sum() / b_air_out_series.sum() - 0.4) < 1e-3
        ), "⚠️ Cold output ratio incorrect"

        # 9️⃣ PV self-consumption increase should be greater than grid import reduction
        # pv_self_use_increase = (
        #     df["pv_self_use_caes"].sum() - df["pv_self_use_ref"].sum()
        # )
        # grid_import_reduction = (
        #     df["grid_import_ref"].sum() - df["grid_import_caes"].sum()
        # )
        # assert (
        #     pv_self_use_increase > grid_import_reduction * 0.4
        # ), "⚠️ PV self-use increase is not as expected"

        #  🔟 **Balance Calculation: Energy Inputs vs. Outputs**
        print("\n📊 **Energy Balance Check** 📊")
        print("-" * 50)

        grid_import = results[(el_source, b_el)]["sequences"]["flow"].sum()
        peak_import = results[(el_peak_source, b_el)]["sequences"]["flow"].sum()
        pv_link_flow = results[(pv_link, b_el)]["sequences"]["flow"].sum()

        total_energy_in = (
            grid_import + peak_import + pv_link_flow + expansion_power.sum()
        )
        total_energy_out = compression_power.sum() + df["demand"].sum()

        print(f"🔹 Grid Import: {grid_import:.2f} kWh")
        print(f"🔹 Peak Import: {peak_import:.2f} kWh")
        print(f"🔹 PV Link Flow: {pv_link_flow:.2f} kWh")
        print(f"🔹 Expansion Power: {expansion_power.sum():.2f} kWh")
        print(f"-----------------------------------")
        print(f"🔸 Compression Power: {compression_power.sum():.2f} kWh")
        print(f"🔸 Demand: {df['demand'].sum():.2f} kWh")
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
    df["grid_import_ref"] = (df["demand"] - df["pv"]).clip(lower=0)
    print("grid_import_ref: ", df["grid_import_ref"].sum())
    print("pv_ref total: ", df["pv"].sum())
    print("demand_ref total: ", df["demand"].sum())

    # Compute PV feed-in (if PV exceeds demand)
    df["pv_feed_in_ref"] = (df["pv"] - df["demand"]).clip(lower=0)
    print("pv_feed_in_ref: ", df["pv_feed_in_ref"].sum())

    # Compute PV self-use as the remaining PV after accounting for feed-in
    df["pv_self_use_ref"] = df["pv"] - df["pv_feed_in_ref"]
    print("pv_self_use_ref: ", df["pv_self_use_ref"].sum())

    # Earnings from using PV directly (self-consumption)
    df["pv_self_use_earnings_ref"] = df["pv_self_use_ref"] * (pv_consumption_compensation / 100)  # Convert to €

    # Earnings from exporting excess PV
    df["pv_feed_in_earnings_ref"] = df["pv_feed_in_ref"] * (feed_in_price / 100)  # Convert to €

    # Grid import cost
    df["cost_grid_import_ref"] = df["grid_import_ref"] * (df["price"] / 100)  # Convert to €

    # peak cost
    peak_cost_ref = df["grid_import_ref"].max() * peak_cost / 100

    # Total cost in the reference case
    df["total_cost_ref"] = (
        df["cost_grid_import_ref"]
        - df["pv_self_use_earnings_ref"]
        - df["pv_feed_in_earnings_ref"]
    )

    # CAES case (optimized system)
    df["grid_import_caes"] = results[(el_source, b_el)]["sequences"]["flow"] + results[(el_peak_source, b_el)]["sequences"]["flow"]
    print("grid_import_caes: ", df["grid_import_caes"].sum())
    df["pv_feed_in_caes"] = results[(b_pv, excess_sink)]["sequences"]["flow"]
    df["compression_power"] = results[(b_el, compression_converter)]["sequences"]["flow"]
    print("compression_power: ", df["compression_power"].sum())
    df["expansion_power"] = results[(expansion_converter, b_el)]["sequences"]["flow"]
    print("expansion_power: ", df["expansion_power"].sum())
    print("^difference: ", df["compression_power"].sum() - df["expansion_power"].sum())
    df["heat_output"] = results[(compression_converter, b_heat)]["sequences"]["flow"]
    print("heat_output: ", df["heat_output"].sum())
    df["cold_output"] = results[(expansion_converter, b_cold)]["sequences"]["flow"]
    print("cold_output: ", df["cold_output"].sum())
    df["pv_self_use_caes"] = (df["pv"] - df["pv_feed_in_caes"]
    )  # PV used directly or stored in the CAES
    print("pv_self_use_caes: ", df["pv_self_use_caes"].sum())

    df["cost_grid_import_caes"] = df["grid_import_caes"] * df["price"] / 100  # Convert to €
    cost_peak_caes = df["grid_import_caes"].max() * peak_cost / 100
    df["pv_self_use_earnings_caes"] = df["pv_self_use_caes"] * pv_consumption_compensation / 100  # Convert to €
    df["pv_feed_in_earnings_caes"] = df["pv_feed_in_caes"] * feed_in_price / 100  # Convert to €

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
    df["pv_feed_in_difference"] = df["pv_feed_in_earnings_caes"] - df["pv_feed_in_earnings_ref"]
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


def plot_results(start_time, end_time, soc, df):
    # PLOTTING
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # First subplot: demand, PV, Price, Battery Power
    ax1 = axes[0]
    grid_import = results[(el_source, b_el)]["sequences"]["flow"]
    peak_grid_import = results[(el_peak_source, b_el)]["sequences"]["flow"]
    max_grid_import = (grid_import + peak_grid_import).max()
    print("max grid import: ", max_grid_import)
    if max_grid_import > peak_threshold:
        peak_time = (grid_import + peak_grid_import).idxmax()
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
            y=max_grid_import + 5,  # Slightly above the line
            s="Real max peak - threshold broken!",
            color="red",
            fontsize=10,
            fontweight="bold",
            ha="center",  # Center align text
        )
    else:
        df_cut = df.loc[start_time:end_time]

    # Filter data to the selected time range
    soc = soc.loc[start_time:end_time]

    demand_results = results[(b_el, demand)]["sequences"]["flow"].loc[
        start_time:end_time
    ]
    pv_results = results[(pv, b_pv)]["sequences"]["flow"].loc[start_time:end_time]

    grid_export = results[(b_pv, excess_sink)]["sequences"]["flow"].loc[
        start_time:end_time
    ]
    compression_power = results[(b_el, compression_converter)]["sequences"]["flow"].loc[
        start_time:end_time
    ]
    expansion_power = results[(expansion_converter, b_el)]["sequences"]["flow"].loc[
        start_time:end_time
    ]
    ax1.axhline(peak_threshold, color="red", linestyle="dashed")
    ax1.plot(demand_results.index, demand_results, label="Demand", color="blue")
    ax1.plot(pv_results.index, pv_results, label="PV Production", color="orange")

    p_g_import_cut = grid_import + peak_grid_import.loc[start_time:end_time]
    ax1.plot(
        p_g_import_cut.index, p_g_import_cut, label="Peak Grid Import", color="red"
    )
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
            print("adjusting price_max: ", price_max)
        else:
            price_min = price_max / power_ratio
            print("adjusting price_min: ", price_min)

        ax1.set_ylim(power_min, power_max)
        ax2.set_ylim(price_min, price_max)

    align_zero_levels(ax1, ax2)

    ax1.set_title("Energy demand, PV Production, Battery Power, and Electricity Price")

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

df = preprocess_df(df, results)
df_re = df.copy()
df_re = preprocess_df(df_re, results)

# Run the recalculation
df_re = recalculate_compression_expansion(df_re, results)

for df in [df, df_re]:
    validate_caes_model(df, results)
    evaluate_economic_impact(df, results)
    plot_results(start_time, end_time, soc, df)
