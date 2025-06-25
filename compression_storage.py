from __future__ import annotations

"""
Complete CAES‑dispatch script – **config‑driven**
====================================================
Only literals that remain are explanatory strings. All technical
parameters, thresholds, file paths and solver options are read from
*conf/config.ini*.
"""

from math import sqrt
import numpy as np
import pandas as pd
import oemof.solph as solph
import matplotlib.pyplot as plt
import sys
import re
import csv
from pyomo.environ import Var, ConstraintList, Binary
from pyomo.opt import SolverFactory
from tabulate import tabulate
from src.color_mapping import assign_colors_to_columns
import configparser
from pathlib import Path
import preprocess_data


# ────────────────────────────────
# 1. Configuration
# ────────────────────────────────
CFG = configparser.ConfigParser()
CFG.read("conf/config.ini", encoding="utf-8")

# general flags
NON_SIMULT = CFG.getboolean("general", "non_simultaneity")
LESS_HEAT_AND_COLD_PRICE = CFG.getboolean("general", "less_heat_and_cold_price")

# peak‑mode / threshold & cost
PEAK_COST_CENT_KW_nopeak = CFG.getfloat("pricing", "cost_eur_per_kw") * 100
if CFG.getboolean("general", "peak_mode"):
    PEAK_THRESHOLD = CFG.getfloat("peak", "threshold_kw_peak")  # kW
    PEAK_COST_CENT_KW = CFG.getfloat("pricing", "cost_eur_per_kw_peak") * 100
else:
    PEAK_THRESHOLD = CFG.getfloat("peak", "threshold_kw")  # kW
    PEAK_COST_CENT_KW = PEAK_COST_CENT_KW_nopeak

# pricing
FEED_IN_PRICE = CFG.getfloat("pricing", "feed_in_price")  # €cent/kWh
PV_COMP_1 = CFG.getfloat("pricing", "pv_consumption_compensation1")  # €cent/kWh
PV_COMP_2 = CFG.getfloat("pricing", "pv_consumption_compensation2")  # €cent/kWh
CONVERTER_COSTS_CENT = CFG.getfloat(
    "pricing", "converter_costs_cent_per_kwh", fallback=0.0
)

# CAES storage parameters
CAES_USAGE = CFG.getboolean("caes_storage", "caes_usage")
CAES_CAPACITY_KWH = CFG.getfloat("caes_storage", "caes_capacity_kwh")
CHARGE_POWER_KW = CFG.getfloat("caes_storage", "storage_charge_power_kw")
DISCHARGE_POWER_KW = CFG.getfloat("caes_storage", "storage_discharge_power_kw")
STORAGE_LOSS_RATE = CFG.getfloat("caes_storage", "storage_loss_rate")

# NACL storage parameters
NACL_USAGE = CFG.getboolean("nacl_storage", "nacl_usage")
NACL_CAPACITY_KWH = CFG.getfloat("nacl_storage", "nacl_capacity_kwh")
NACL_POWER_KW = CFG.getfloat("nacl_storage", "nacl_storage_charge_power_kw")
ETA_NACL =  CFG.getfloat("nacl_storage", "eta_nacl")

# cold storage parameters
COLD_CAPACITY_KWH = CFG.getfloat("cold_storage", "cold_capacity_kwh")

# time‑parameters
START_TIME = CFG["time"]["start_time"]
END_TIME = CFG["time"]["end_time"]
FREQ = CFG["time"]["resample_frequency"]

# paths
DATA_CSV = Path(CFG["paths"]["cleaned_data_csv"])
STORAGE_RESULTS_CSV = Path(CFG["paths"]["storage_results_csv"])
PREPROCESSED_CSV = Path(CFG["paths"]["preprocessed_results_csv"])
ENERGY_BALANCE_CSV = Path(CFG["paths"]["energy_balance_csv"])
ECONOMIC_SUM_CSV = Path(CFG["paths"]["economic_summary_csv"])

# solver opts
SOLVER_NAME = CFG["solver"]["name"]
SOLVER_TEE = CFG.getboolean("solver", "tee")
SOLVER_GAP = CFG.getfloat("solver", "ratio_gap")

# tiered PV fraction
TIER_FRACTION = CFG.getfloat("tiered_pv", "threshold_fraction")


# Define column names for reference and CAES results
ref_energy_columns = ["demand", "pv", "pv_feed_in_ref", "grid_import_ref"]
caes_energy_columns = [
    "demand",
    "pv",
    "pv_feed_in_caes",
    "grid_import_caes",
    "compression_power",
    "expansion_power",
    "heat_output",
    "cold_chiller",
    "cold_freezer",
]

ref_cost_columns = [
    "cost_grid_import_ref",
    "peak_cost_ref",
    "pv_feed_in_earnings_ref",
    "pv_self_use_earnings_ref",
]
caes_cost_columns = [
    "cost_grid_import_caes",
    "peak_cost_caes",
    "pv_feed_in_earnings_caes",
    "pv_self_use_earnings_caes",
    "heat_earnings_caes",
    "cold_earnings_chiller",
    "cold_earnings_freezer",
    "compression_cost",
    "expansion_cost",
]


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


heat_price = df["heat_price"]
q_demand_chiller = df["Q_demand_chiller"]  # kWh/hour
q_max_chiller = df["Q_demand_chiller"].max()
q_demand_freezer = df["Q_demand_freezer"]  # kWh/hour
q_max_freezer = df["Q_demand_freezer"].max()
cold_price_chiller = df["cold_price_chiller"]  # €cent/kWh
cold_price_freezer = df["cold_price_freezer"]  # €cent/kWh

if LESS_HEAT_AND_COLD_PRICE:
    factor_heat = 0.55
else:
    factor_heat = 1

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
            min=0, nominal_value=1000, variable_costs=-PV_COMP_2
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
    label="excess_sink", inputs={b_pv: solph.flows.Flow(variable_costs=-FEED_IN_PRICE)}
)
energy_system.add(excess_sink)

# Grid source with variable electricity prices
el_source = solph.components.Source(
    label="el_source",
    outputs={
        b_el: solph.flows.Flow(
            nominal_value=PEAK_THRESHOLD, variable_costs=df["price"]
        )
    },
)
energy_system.add(el_source)

# Peak electricity source with higher variable costs
el_peak_source = solph.components.Source(
    label="el_peak_source",
    outputs={b_el: solph.flows.Flow(variable_costs=df["price"] + PEAK_COST_CENT_KW)},
)
energy_system.add(el_peak_source)

# Electricity demand as a sink
demand = solph.components.Sink(
    label="demand_sink",
    inputs={b_el: solph.flows.Flow(fix=df["demand"], nominal_value=1)},
)
energy_system.add(demand)

if CAES_USAGE:
    # storage system
    storage = solph.components.GenericStorage(
        label="storage",
        inputs={b_air: solph.flows.Flow(nominal_value=100)},  # 100 kW charge power
        outputs={b_air: solph.flows.Flow(nominal_value=100)},  # 100 kW discharge power
        nominal_storage_capacity=CAES_CAPACITY_KWH,  # kWh total storage capacity
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
        inputs={
            b_el: solph.flows.Flow(nominal_value=100, variable_costs=CONVERTER_COSTS_CENT)
        },  # Max input power 100 kW
        outputs={
            b_air: solph.flows.Flow(nominal_value=100),  # Storing compressed air
            b_heat: solph.flows.Flow(nominal_value=90),  # Extracting heat
        },
        conversion_factors={
            b_air: 1,  # 100 kWh of electricity goes into 100 kWh compressed air
            b_heat: 0.9,  # 90 kWh heat extracted during compression
        },
    )
    energy_system.add(compression_converter)

    # Expansion Process: Compressed Air → Electricity + Cold
    expansion_converter = solph.components.Converter(
        label="expansion_converter",
        inputs={
            b_air: solph.flows.Flow(nominal_value=100, variable_costs=CONVERTER_COSTS_CENT)
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

    # Create the cold storage with a capacity of 83 kWh
    cold_storage = solph.components.GenericStorage(
        label="cold_storage",
        inputs={b_cold: solph.flows.Flow()},
        outputs={b_cold: solph.flows.Flow()},
        nominal_storage_capacity=COLD_CAPACITY_KWH,  # in kWh
        initial_storage_level=0.5,  # for instance, starting full; adjust as needed
        loss_rate=0.0,  # adjust if there are standing losses
        inflow_conversion_factor=1,
        outflow_conversion_factor=1,
        balanced=True,
    )
    energy_system.add(cold_storage)

    # sink for heat
    heat_sink = solph.components.Sink(
        label="heat_sink",
        inputs={b_heat: solph.flows.Flow(variable_costs=-heat_price * factor_heat)},
    )
    energy_system.add(heat_sink)

    # Create two cold sinks
    # For freezer cooling
    cold_sink_freezer = solph.components.Sink(
        label="cold_sink_freezer",
        inputs={
            b_cold: solph.flows.Flow(
                variable_costs=-cold_price_freezer * factor_heat,
                nominal_value=q_max_freezer,
                max=q_demand_freezer / q_max_freezer,
            )
        },
    )
    energy_system.add(cold_sink_freezer)

    # For chiller cooling
    cold_sink_chiller = solph.components.Sink(
        label="cold_sink_chiller",
        inputs={
            b_cold: solph.flows.Flow(
                variable_costs=-cold_price_chiller * factor_heat,
                nominal_value=q_max_chiller,
                max=q_demand_chiller / q_max_chiller,
            )
        },
    )
    energy_system.add(cold_sink_chiller)

    cold_sink = solph.components.Sink(
        label="cold_sink", inputs={b_cold: solph.flows.Flow()}
    )
    energy_system.add(cold_sink)
elif NACL_USAGE:
    storage = solph.components.GenericStorage(
        label="battery",
        inputs={b_el: solph.flows.Flow(nominal_value=NACL_POWER_KW)},  # kW charge
        outputs={b_el: solph.flows.Flow(nominal_value=NACL_POWER_KW)},  # kW discharge
        nominal_storage_capacity=NACL_CAPACITY_KWH,
        initial_storage_level=0.5,
        loss_rate=0,  # set your calendar losses if any
        balanced=True,
        inflow_conversion_factor=sqrt(ETA_NACL),
        outflow_conversion_factor=sqrt(ETA_NACL),
    )
    energy_system.add(storage)


# Create and solve the optimization model
model = solph.Model(energy_system)


def apply_non_simultaneity_constraints(
    model, compression_converter, expansion_converter, b_air, enable=True
):
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
            model.flow[compression_converter, b_air, t] <= M * model.non_simul_mode[t]
        )
        # Constraint: If non_simul_mode[t] is 0, expansion flow can be up to M, but compression must be 0.
        model.non_simul_constraints.add(
            model.flow[b_air, expansion_converter, t]
            <= M * (1 - model.non_simul_mode[t])
        )


if CAES_USAGE:
    # Apply the non-simultaneity constraints
    apply_non_simultaneity_constraints(
        model, compression_converter, expansion_converter, b_air, enable=NON_SIMULT
    )

# Solve the optimization model
model.solve(solver="cbc", solve_kwargs={"tee": True, "options": {"ratioGap": 0.001}})

# Extract results
results = solph.processing.results(model)

# print("Results keys:")
# for k in results.keys():
#     print(k)  # See what keys exist

meta_results = solph.processing.meta_results(model)

# Convert results to DataFrame for analysis
storage_flows = results[(storage, None)]["sequences"]

# Save results
storage_flows.to_csv("storage_results.csv")

# Print key results
print(meta_results)
print(storage_flows.head())

# Define start and end time for the plot
start_time = "2024-01-01"
end_time = "2024-12-31"

subtract_columns = [
    "demand",
    "compression_power",
    "pv_feed_in_earnings",
    "pv_self_use_earnings",
    "heat_earnings",
    "cold_earnings_chiller",
    "cold_earnings_freezer",
]


def preprocess_ref(df):
    df["grid_import_ref"] = (df["demand"] - df["pv"]).clip(lower=0)
    df["pv_feed_in_ref"] = (df["pv"] - df["demand"]).clip(lower=0)
    df["pv_self_use_ref"] = df["pv"] - df["pv_feed_in_ref"]
    return df


def preprocess_caes(df, results):
    # Transfer all relevant caes result flows into df columns.
    # --- original result series -----------------------------------------------
    grid_import_orig = (
        results[(el_source, b_el)]["sequences"]["flow"]
        + results[(el_peak_source, b_el)]["sequences"]["flow"]
    ).loc[df.index]
    pv_feed_orig = results[(b_pv, excess_sink)]["sequences"]["flow"].loc[df.index]

    overlap = np.minimum(grid_import_orig, pv_feed_orig)
    print("total overlap: ", overlap.sum())

    # --- unidirectional correction --------------------------------------------
    diff            = grid_import_orig - pv_feed_orig
    grid_corrected  = diff.clip(lower=0)
    df["grid_import_caes"] = grid_corrected
    pv_corrected    = (-diff).clip(lower=0)
    df["pv_feed_in_caes"] = pv_corrected

    # --- overlap (= “correction”) ---------------------------------------------
    # overlap = grid_import_orig - grid_corrected          # identical to pv_feed_orig - pv_corrected
    # or, equivalently:
    #

    # plot the difference with related timeseries
    # plt.plot(df.index, grid_import_orig, label="grid_import_caes")
    # plt.plot(df.index, pv_feed_orig, label="pv_feed_in_caes")
    # # plt.plot(df.index, df["grid_import_caes2"], label="grid_import_caes2")
    # # plt.plot(df.index, df["pv_feed_in_caes2"], label="pv_feed_in_caes2")
    # plt.plot(df.index, diff, label="difference", alpha=0.5)
    # plt.plot(df.index, overlap, label="overlap")
    # plt.axhline(0, color="black", lw=0.5, ls="--")
    # plt.title("Difference between grid import and pv feed-in")
    # plt.xlabel("Time")
    # plt.ylabel("kW")
    # plt.legend()
    # plt.grid()
    # plt.show()

    df["pv_self_use_caes"] = df["pv"] - df["pv_feed_in_caes"]
    if CAES_USAGE:
        df["compression_power"] = results[(b_el, compression_converter)]["sequences"][
            "flow"
        ].loc[df.index]
        df["expansion_power"] = results[(expansion_converter, b_el)]["sequences"][
            "flow"
        ].loc[df.index]
        df["heat_output"] = results[(compression_converter, b_heat)]["sequences"][
            "flow"
        ].loc[df.index]
        # df["cold_output"] = results[(b_cold, cold_sink_chiller)]["sequences"]["flow"].loc[df.index] \
        #     + results[(b_cold, cold_sink_freezer)]["sequences"]["flow"].loc[df.index]
        df["cold_chiller"] = results[(b_cold, cold_sink_chiller)]["sequences"]["flow"].loc[df.index]
        df["cold_freezer"] = results[(b_cold, cold_sink_freezer)]["sequences"]["flow"].loc[df.index]
        df["cold_waste"] = results[(b_cold, cold_sink)]["sequences"]["flow"].loc[df.index]
        df["soc"] = results[(storage, None)]["sequences"]["storage_content"].loc[df.index]
        df["soc_cold"] = results[(cold_storage, None)]["sequences"]["storage_content"].loc[df.index]
    elif NACL_USAGE:
        df["compression_power"] = results[(b_el, storage)]["sequences"]["flow"].loc[df.index].clip(lower=0)
        df["expansion_power"] = (
            results[(storage, b_el)]["sequences"]["flow"].loc[df.index].clip(lower=0)
        )
        df["heat_output"] = 0.0
        df["cold_chiller"] = 0.0
        df["cold_freezer"] = 0.0
        df["cold_waste"] = 0.0
        df["soc"] = results[(storage, None)]["sequences"]["storage_content"].loc[
            df.index
        ]
        df["soc_cold"] = 0.0
    return df


def save_preprocessed_df(df):
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
    df["expansion_power"] = (
        -soc_change.clip(upper=0) * expansion_efficiency
    )  # Only negative changes = expansion

    # Compute heat and cold outputs
    heat_output_efficiency = 0.9  # heat recovery from compression
    cold_output_efficiency = 0.4  # cooling output from expansion

    df["heat_output"] = df["compression_power"] * heat_output_efficiency
    # df["cold_output"] = df["expansion_power"] * cold_output_efficiency

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
        assert (
            df["pv_self_use_caes"].min() >= -1e-3
        ), f"⚠️ PV link flow should always be positive but is {df['pv_self_use_caes'].min()} kWh in timestep {df['pv_self_use_caes'].idxmin()}"

        # 4️⃣ Only PV should feed the excess sink
        pv_production = results[(pv, b_pv)]["sequences"]["flow"]
        excess_sink_series = results[(b_pv, excess_sink)]["sequences"]["flow"]
        assert (
            pv_production - excess_sink_series
        ).min() >= 0, "⚠️ Only PV should feed the excess sink"

        # 5️⃣ Stored compressed air should match expanded air
        b_air_in_series = results[(compression_converter, b_air)]["sequences"]["flow"]
        b_air_out_series = results[(b_air, expansion_converter)]["sequences"]["flow"]
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
        cold_output = (
            df["cold_chiller"].sum()
            + df["cold_freezer"].sum()
            + df["cold_waste"].sum()
        )
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
            abs(
                df["pv_self_use_caes"].sum()
                + df["pv_feed_in_caes"].sum()
                - df["pv"].sum()
            )
            < 1e-3
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


def tiered_pv_earnings(pv_series, pv_self_use_series):
    # Total PV generation and self-consumption over the year
    pv_gen_total = pv_series.sum()
    pv_self_use_total = pv_self_use_series.sum()

    # Compensation thresholds
    threshold_kwh = 0.30 * pv_gen_total

    # Split self-consumption into tiers
    tier1_kwh = min(pv_self_use_total, threshold_kwh)
    tier2_kwh = max(pv_self_use_total - threshold_kwh, 0)

    # Earnings
    earnings_self_use = (tier1_kwh * PV_COMP_1 / 100 +
                         tier2_kwh * PV_COMP_2 / 100)

    # Distribute annual earnings proportionally across time steps
    return pv_self_use_series / pv_self_use_total * earnings_self_use


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
    df["pv_self_use_earnings_ref"] = 0
    df["pv_self_use_earnings_ref"] = tiered_pv_earnings(df["pv"], df["pv_self_use_ref"])

    # Earnings from exporting excess PV
    df["pv_feed_in_earnings_ref"] = df["pv_feed_in_ref"] * (
        FEED_IN_PRICE / 100
    )  # Convert to €

    # Grid import cost
    df["cost_grid_import_ref"] = df["grid_import_ref"] * (
        df["price_nopeak"] / 100
    )  # Convert to €

    # peak cost reference
    peak_time_ref = df["grid_import_ref"].idxmax()
    peak_cost_ref = df["grid_import_ref"].max() * PEAK_COST_CENT_KW_nopeak / 100
    print("peak cost ref: ", peak_cost_ref)
    df["peak_cost_ref"] = 0.0  # explizit float
    df.loc[:, "peak_cost_ref"] = 0.0
    # df["peak_cost_ref"] = 0
    df.at[peak_time_ref, "peak_cost_ref"] = peak_cost_ref

    # Total cost in the reference case
    df["total_cost_ref"] = (
        df["cost_grid_import_ref"]
        + df["peak_cost_ref"]
        - df["pv_self_use_earnings_ref"]
        - df["pv_feed_in_earnings_ref"]
    )

    # CAES case (optimized system)
    print("grid_import_caes: ", df["grid_import_caes"].sum())
    print("compression_power: ", df["compression_power"].sum())
    print("expansion_power: ", df["expansion_power"].sum())
    # print("^difference: ", df["compression_power"].sum() - df["expansion_power"].sum())
    print("heat_output: ", df["heat_output"].sum())
    print("cold_chiller output: ", df["cold_chiller"].sum())
    print("cold_freezer output: ", df["cold_freezer"].sum())

    # daily average and max cold output
    daily_avg_cold_output = df[["cold_chiller", "cold_freezer"]].resample("D").sum().mean().sum()
    daily_max_cold_output = df[["cold_chiller", "cold_freezer"]].resample("D").sum().max().sum()
    print("daily average cold used: ", daily_avg_cold_output)
    print("daily max cold used: ", daily_max_cold_output)

    print("pv_self_use_caes: ", df["pv_self_use_caes"].sum())

    df["cost_grid_import_caes"] = (
        df["grid_import_caes"] * df["price"] / 100
    )  # Convert to €

    peak_time_caes = df["grid_import_caes"].idxmax()
    peak_cost_caes = df["grid_import_caes"].max() * PEAK_COST_CENT_KW / 100
    df["peak_cost_caes"] = 0
    df.at[peak_time_caes, "peak_cost_caes"] = peak_cost_caes

    df["pv_self_use_earnings_caes"] = tiered_pv_earnings(df["pv"], df["pv_self_use_caes"])
    df["pv_feed_in_earnings_caes"] = (
        df["pv_feed_in_caes"] * FEED_IN_PRICE / 100
    )  # Convert to €

    df["heat_earnings_caes"] = df["heat_output"] * heat_price / 100
    df["cold_earnings_chiller"] = df["cold_chiller"] * cold_price_chiller / 100
    df["cold_earnings_freezer"] = df["cold_freezer"] * cold_price_freezer / 100

    df["total_cost_caes"] = (
        df["cost_grid_import_caes"]
        + df["peak_cost_caes"]
        - df["pv_self_use_earnings_caes"]
        - df["pv_feed_in_earnings_caes"]
        - df["heat_earnings_caes"]
        - df["cold_earnings_chiller"]
        - df["cold_earnings_freezer"]
    )

    # Economic impact of CAES
    df["cost_savings"] = df["total_cost_ref"] - df["total_cost_caes"]
    df["grid_import_reduction"] = df["grid_import_ref"] - df["grid_import_caes"]
    df["pv_self_use_increase"] = df["pv_self_use_caes"] - df["pv_self_use_ref"]
    df["pv_feed_in_difference"] = (
        df["pv_feed_in_earnings_caes"] - df["pv_feed_in_earnings_ref"]
    )
    peak_cost_difference = peak_cost_ref - peak_cost_caes
    cost_savings = df["cost_savings"].sum() + peak_cost_difference

    print("\n### Economic Impact of CAES ###")
    print(f"Total cost savings: {cost_savings:.2f} €")
    print(f"Thereof peak cost difference: {peak_cost_difference:.2f} €")
    print(f"Grid import reduction: {df['grid_import_reduction'].sum():.2f} kWh")
    print(f"PV self-consumption increase: {df['pv_self_use_increase'].sum():.2f} kWh")
    print(f"PV feed-in revenue change: {df['pv_feed_in_difference'].sum():.2f} €")
    print(f"Heat earnings: {df['heat_earnings_caes'].sum():.2f} €")
    print(f"Cold earnings chiller: {df['cold_earnings_chiller'].sum():.2f} €")
    print(f"Cold earnings freezer: {df['cold_earnings_freezer'].sum():.2f} €")

    # Warning if CAES increases cost instead of reducing it
    if df["cost_savings"].sum() < 0:
        print(
            "\n⚠️ Warning: CAES increased total costs instead of reducing them! Review input assumptions."
        )


def calculate_storage_cycles_simple(soc_series, storage_capacity):
    """
    Calculates storage cycles by summing all charge events and dividing by capacity.

    Parameters:
    - soc_series (pd.Series): Time series of state of charge (SOC).
    - storage_capacity (float): Maximum storage capacity (kWh).

    Returns:
    - total_cycles (float): Number of full storage cycles.
    """

    # Compute SOC change
    soc_diff = soc_series.diff().fillna(0)

    # Sum only positive changes (charging events)
    total_charge_energy = soc_diff[soc_diff > 0].sum()

    # Compute full cycle equivalent
    total_cycles = total_charge_energy / storage_capacity

    return total_cycles


def calculate_monthly_storage_stats(soc_series):
    """
    Calculate storage usage statistics for each month.

    Parameters:
    - soc_series (pd.Series): Time series of state of charge (SOC).
    - storage_capacity (float): Maximum storage capacity (kWh).

    Returns:
    - storage_stats (pd.DataFrame): Monthly storage cycle count and DOD.
    """

    # Assume full capacity is the maximum SOC value
    storage_capacity = soc_series.max()
    print("storage_capacity: ", storage_capacity)

    monthly_stats = []
    for month, soc in soc_series.resample("ME"):
        if soc.empty:
            continue

        # Count full & partial cycles using the simple sum-based method
        cycles = calculate_storage_cycles_simple(soc, storage_capacity)

        # Calculate active hours (time with non-zero SOC change)
        active_hours = (soc.diff().abs() > 0).sum()

        # Calculate average monthly depth of discharge (DOD)
        # dod_values = soc.diff().abs().dropna()
        # avg_dod = dod_values.mean() if not dod_values.empty else 0

        monthly_stats.append(
            {
                "month": month,
                "usage_cycles": cycles,
                "active_hours": active_hours,
                # "avg_dod": avg_dod,
            }
        )
        # print
        print(f"Month: {month.strftime('%B %Y')}")
        print(f"  - Full Cycles: {cycles:.2f}")
        print(f"  - Active Hours: {active_hours}")
        # print(f"  - Avg. DOD: {avg_dod:.2f} kWh")

    # total cycles
    total_cycles = sum([entry["usage_cycles"] for entry in monthly_stats])
    print(f"Total Cycles: {total_cycles:.2f}")

    storage_stats = pd.DataFrame(monthly_stats).set_index("month")
    return storage_stats


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


def plot_hourly_resolution_energy_flows_ref(start_time, end_time, df):
    """
    Plot hourly energy flows for the Reference scenario only.
    """
    # Set up figure (using same gridspec for consistency)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Define the reference columns (example)
    ref_columns = ["demand", "pv", "pv_feed_in_ref", "grid_import_ref"]
    ref_colors = assign_colors_to_columns(ref_columns)

    # Resample data if needed; here we assume df is already hourly.
    df_cut = df.loc[start_time:end_time]

    # Plot each reference column using the external color mapping
    for i, col in enumerate(ref_columns):
        ax1.plot(
            df_cut.index,
            df_cut[col],
            label=f"Ref - {col}",
            color=ref_colors[i] if i < len(ref_colors) else "black",
        )

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Power [kW]")
    ax1.set_title("Hourly Energy Flows - Reference Scenario")
    ax1.legend(loc="upper left")

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

    # align_zero_levels(ax1, ax2)

    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()


def plot_hourly_resolution_energy_flows(start_time, end_time, df):

    # Define colors for the Reference and CAES columns
    # ref_colors = assign_colors_to_columns(ref_energy_columns)
    caes_colors = assign_colors_to_columns(caes_energy_columns)
    # create a dictionary with the columns and colors
    columns_colors = dict(zip(caes_energy_columns, caes_colors))

    # caes_energy_columns = [
    #     "demand",
    #     "pv",
    #     "pv_feed_in_caes",
    #     "grid_import_caes",
    #     "compression_power",
    #     "expansion_power",
    #     "heat_output",
    #     "cold_output",
    # ]

    # PLOTTING
    # fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # First subplot: demand, PV, Price, CAES Power
    ax1 = axes[0]
    grid_import = df["grid_import_caes"]
    max_grid_import = grid_import.max()
    print("max grid import: ", max_grid_import)

    if max_grid_import > PEAK_THRESHOLD:
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

    ax1.axhline(PEAK_THRESHOLD, color="red", linestyle="dashed")
    ax1.plot(demand_results.index, demand_results, label="Demand", color=columns_colors["demand"])
    ax1.plot(pv_results.index, pv_results, label="PV Production", color=columns_colors["pv"])

    grid_import_cut = grid_import.loc[start_time:end_time]
    ax1.plot(grid_import_cut.index, grid_import_cut, label="Grid Import", color=columns_colors["grid_import_caes"])

    ax1.plot(grid_export.index, -grid_export, label="Grid Export", color=columns_colors["pv_feed_in_caes"])
    label_charge = "Compression Power" if CAES_USAGE else "Charge Power"
    label_discharge = "Expansion Power" if CAES_USAGE else "Discharge Power"
    ax1.plot(
        compression_power.index,
        -compression_power,
        label=label_charge,
        color=columns_colors["compression_power"],
    )
    ax1.plot(
        expansion_power.index, expansion_power, label=label_discharge, color=columns_colors["expansion_power"]
    )

    # pv_link_flow = results[(pv_link, b_el)]["sequences"]["flow"]
    # ax1.plot(pv_link_flow.index, pv_link_flow, label="PV Link Flow", color="brown")

    ax1.set_ylabel("Power [kW]")
    ax1.legend(loc="lower left")
    # Add a single horizontal helper line at zero
    ax1.axhline(0, color="black", linestyle="dashed")

    # ✅ **Force x-axis labels to be shown on ax1**
    ax1.xaxis.set_tick_params(labelbottom=True)

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

    align_zero_levels(ax1, ax2)

    ax1.set_title("Energy demand, PV Production, CAES Power, and Electricity Price")

    # Second subplot: State of Charge (SOC)
    ax3 = axes[1]
    ax3.plot(df_cut.index, df_cut["soc"], label="CAES SOC", color="purple")
    ax3.plot(df_cut.index, df_cut["soc_cold"], label="Cold SOC", color="blue")
    ax3.set_ylabel("State of Charge [kWh]")
    # ax3.axhline(0, color="gray", linestyle="dashed")  # Horizontal helper line for SOC
    # ax3.spines["left"].set_position(("data", 0))  # Move y-axis to cross x-axis at zero
    ax3.legend()
    ax3.set_title("CAES State of Charge")

    # Align y-axis zero levels
    ax1.set_ylim(min(ax1.get_ylim()[0], 0), ax1.get_ylim()[1])
    ax2.set_ylim(min(ax2.get_ylim()[0], 0), ax2.get_ylim()[1])
    ax3.set_ylim(min(ax3.get_ylim()[0], 0), ax3.get_ylim()[1])

    # Final formatting
    axes[1].set_xlabel("Time")
    ax3.xaxis.set_tick_params(labelbottom=True)  # Show x-axis labels on both subplots
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()


def plot_cost_series(df):
    """
    Plot cost series for the reference and CAES case in a single figure with consistent colors.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 🎨 **Define Color Mapping**
    colors = {
        "total_cost": "black",
        "grid_import": "gray",
        "pv_feed_in": "goldenrod",
        "pv_self_use": "orange",
        "heat_earnings": "red",
        "cold_earnings_chiller": "blue",
        "cold_earnings_freezer": "cyan",
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
        -df["cold_earnings_chiller"].cumsum(),
        label="Cold Earnings Chiller (CAES)",
        color=colors["cold_earnings_chiller"],
    )
    ax.plot(
        df.index,
        -df["cold_earnings_freezer"].cumsum(),
        label="Cold Earnings Freezer (CAES)",
        color=colors["cold_earnings_freezer"],
    )
    ax.plot(
        df.index,
        (df["compression_power"] * CONVERTER_COSTS_CENT / 100).cumsum(),
        label="Compression Costs (CAES)",
        color=colors["compression_cost"],
    )
    ax.plot(
        df.index,
        (df["expansion_power"] * CONVERTER_COSTS_CENT / 100).cumsum(),
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
    # plt.show()


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
            df["heat_output"].sum(),
            df["cold_chiller"].sum() + df["cold_freezer"].sum(),
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
        "(Heat Used)",
        "(Cold Used)",
        "Total Electric Energy In",
        "Total Electric Energy Out",
        "Energy Balance (In - Out)",
    ]

    df_energy = pd.DataFrame(balance_data, index=index)
    df_energy.to_csv("results/energy_balance.csv")
    print(tabulate(df_energy, headers="keys", tablefmt="fancy_grid", floatfmt=".2f"))


def create_economic_summary_table(df):
    """
    Create a summary table for the economic impact of the CAES system.
    """

    def define_peak_costs(df, column):
        peak_time = df[column].idxmax()
        peak_cost_value = df[column].max() * PEAK_COST_CENT_KW / 100
        column_name = "peak_cost_" + column.split("_")[-1]
        df[column_name] = 0.0
        df.loc[:, column_name] = 0.0
        df.at[peak_time, column_name] = peak_cost_value

    # Peak costs are already calculated in the evaluate_economic_impact function
    # define_peak_costs(df, "grid_import_ref")
    # define_peak_costs(df, "grid_import_caes")

    df["compression_cost"] = df["compression_power"] * CONVERTER_COSTS_CENT / 100
    df["expansion_cost"] = df["expansion_power"] * CONVERTER_COSTS_CENT / 100

    economics_data = {
        "Reference [€]": [
            df["cost_grid_import_ref"].sum(),
            df["peak_cost_ref"].sum(),
            -df["pv_feed_in_earnings_ref"].sum(),
            -df["pv_self_use_earnings_ref"].sum(),
            0,
            0,
            0,
            0,
            0,
            df["total_cost_ref"].sum(),
        ],
        "CAES [€]": [
            df["cost_grid_import_caes"].sum(),
            df["peak_cost_caes"].sum(),
            -df["pv_feed_in_earnings_caes"].sum(),
            -df["pv_self_use_earnings_caes"].sum(),
            -df["heat_earnings_caes"].sum(),
            -df["cold_earnings_chiller"].sum(),
            -df["cold_earnings_freezer"].sum(),
            df["compression_cost"].sum(),
            df["expansion_cost"].sum(),
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
            "Cold Earnings Chiller",
            "Cold Earnings Freezer",
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


def adjust_columns_for_plotting(df, subtract_columns):
    df_adj = df.copy()
    for col in df_adj.columns:
        # Remove _caes or _ref if present
        base_col = re.sub(r"(_caes|_ref)$", "", col)  # Remove ONLY full suffixes

        # flip sign for relevant columns
        if base_col in subtract_columns or "earning" in col:
            df_adj[col] = -df_adj[col]
    return df_adj


def plot_energy_economics(
    df,
    ref_columns,
    caes_columns,
    resolution="monthly",
    plot_type="energy"
):
    """
    Generalized function to plot energy or economic data over time for both Reference and CAES scenarios.

    Parameters:
    - df: Pandas DataFrame with a DateTime index
    - ref_columns: List of column names for the Reference scenario
    - caes_columns: List of column names for the CAES scenario
    - resolution: "yearly", "monthly", or "daily"
    - plot_type: "energy" or "costs"
    - stacked: Whether to use stacked bars (ignored for daily line plots)
    """

    df_plot = adjust_columns_for_plotting(df, subtract_columns)

    # Resample data based on the chosen resolution
    if resolution == "yearly":
        df_resampled_ref = df_plot[ref_columns].resample("YE").sum()
        df_resampled_caes = df_plot[caes_columns].resample("YE").sum()
        xlabel = "Year"
    elif resolution == "monthly":
        df_resampled_ref = df_plot[ref_columns].resample("ME").sum()
        df_resampled_caes = df_plot[caes_columns].resample("ME").sum()
        xlabel = "Month"
    elif resolution == "daily":
        df_resampled_ref = df_plot[ref_columns].resample("D").sum()
        df_resampled_caes = df_plot[caes_columns].resample("D").sum()
        xlabel = "Day"
    else:
        raise ValueError(
            "Invalid resolution. Choose from 'yearly', 'monthly', 'daily'."
        )

    # Convert index to string format for better readability on x-axis
    if resolution == "monthly":
        df_resampled_ref.index = df_resampled_ref.index.strftime("%Y-%m")
        df_resampled_caes.index = df_resampled_caes.index.strftime("%Y-%m")
    elif resolution == "yearly":
        df_resampled_ref.index = df_resampled_ref.index.strftime("%Y")
        df_resampled_caes.index = df_resampled_caes.index.strftime("%Y")
    else:  # daily case
        df_resampled_ref.index = df_resampled_ref.index.strftime("%Y-%m-%d")
        df_resampled_caes.index = df_resampled_caes.index.strftime("%Y-%m-%d")

    # Define colors for the Reference and CAES columns
    ref_colors = assign_colors_to_columns(ref_columns)
    caes_colors = assign_colors_to_columns(caes_columns)

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set plot type: Bar plot for yearly, monthly, and daily resolutions
    x = np.arange(len(df_resampled_ref.index))  # X locations
    width = 0.4  # Bar width

    def plot_stacked_bars(ax, x, width, df_resampled, columns, colors, label_prefix, shift, set_labels=True):
        """
        Plots stacked bar charts with separate handling for positive and negative values.

        Parameters:
        - ax: Matplotlib axis object.
        - x: Array of x positions.
        - width: Bar width.
        - df_resampled: Resampled DataFrame for plotting.
        - columns: Columns to plot.
        - colors: Assigned colors for columns.
        - label_prefix: "Ref" or "CAES" for labeling.
        - shift: X shift for bar alignment.
        """

        # Initialize cumulative sums for stacking
        cumulative_pos = np.zeros(len(df_resampled))  # Tracks positive stacking
        cumulative_neg = np.zeros(len(df_resampled))  # Tracks negative stacking

        # Plot bars
        for i, col in enumerate(columns):
            values = df_resampled[col].values  # Extract column values

            # Determine where values are positive and negative
            pos_values = np.where(values > 0, values, 0)
            neg_values = np.where(values < 0, values, 0)

            # Plot positive values stacked on cumulative_pos (add legend only for first stacked element)
            ax.bar(
                [j + shift for j in x],
                pos_values,
                width,
                bottom=cumulative_pos,
                label=re.sub(r"(_caes|_ref)$", "", col) if set_labels else None,
                color=colors[i] if i < len(colors) else "black",
            )

            # Plot negative values stacked on cumulative_neg
            ax.bar(
                [j + shift for j in x],
                neg_values,
                width,
                bottom=cumulative_neg,
                # label=f"{label_prefix} - {col}" if i == 0 else None,
                color=colors[i] if i < len(colors) else "black",
            )

            # Update cumulative sums
            cumulative_pos += pos_values
            cumulative_neg += neg_values  # These are negative, so they stack downward

        # Set vertical offset for labels
        offset = 0.5

        # **Add Labels Above Stacked Bars**
        for i, x_pos in enumerate(x):
            total_value = cumulative_pos[i] if cumulative_pos[i] > 0 else cumulative_neg[i]
            ax.text(
                x_pos + shift,
                total_value + offset,  # Place label above top of stacked bar
                label_prefix,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Plot bars for Reference and CAES columns
    plot_stacked_bars(ax, x, width, df_resampled_ref, ref_columns, ref_colors, "Ref", shift=+width, set_labels=False)
    plot_stacked_bars(ax, x, width, df_resampled_caes, caes_columns, caes_colors, "CAES", shift=0, set_labels=True)

    ax.set_xticks([j + width/2 for j in x])
    ax.set_xticklabels(df_resampled_ref.index, rotation=45)

    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy [kWh]" if plot_type == "energy" else "Costs [€]")
    ax.set_title(
        f"Stacked {resolution.capitalize()} {plot_type.capitalize()} Summary"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot at the end of the script
    plt.tight_layout()


def print_energy_economics(df, ref_columns, caes_columns, print_type):

    df_resampled_ref = df[ref_columns].resample("ME").sum()
    df_resampled_caes = df[caes_columns].resample("ME").sum()

    # Convert the resampled indices to the month name (e.g. "April")
    df_resampled_ref.index = df_resampled_ref.index.strftime("%B")
    df_resampled_caes.index = df_resampled_caes.index.strftime("%B")

    # After resetting the index and renaming the column, format numeric values with zero decimals:
    df_ref_table = df_resampled_ref.copy()
    df_ref_table.loc["Total"] = df_ref_table.sum()
    df_ref_table = df_ref_table.reset_index().rename(columns={"index": "Month"})
    for col in df_ref_table.columns[1:]:
        df_ref_table[col] = df_ref_table[col].map(lambda x: f"{x:,.0f}")

    print("Monthly Values - Reference Scenario:")
    print(tabulate(df_ref_table, headers="keys", tablefmt="fancy_grid", showindex=False))

    df_caes_table = df_resampled_caes.copy()
    df_caes_table.loc["Total"] = df_caes_table.sum()
    df_caes_table = df_caes_table.reset_index().rename(columns={"index": "Month"})
    for col in df_caes_table.columns[1:]:
        df_caes_table[col] = df_caes_table[col].map(lambda x: f"{x:,.0f}")

    print("\nMonthly Values - CAES Scenario:")
    print(tabulate(df_caes_table, headers="keys", tablefmt="fancy_grid", showindex=False))

    # df_ref_table.to_csv(f"monthly_{print_type}_values_reference.csv", index=False)
    # df_caes_table.to_csv(f"monthly_{print_type}_values_caes.csv", index=False)

    df_ref_table.to_csv(
        f"monthly_{print_type}_values_reference.csv",
        index=False,
        sep=";",
        decimal=",",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    df_caes_table.to_csv(
        f"monthly_{print_type}_values_caes.csv",
        index=False,
        sep=";",
        decimal=",",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )


def plot_storage_usage_stats(storage_stats):
    """
    Plot storage usage frequency, active hours, and depth of discharge with monthly resolution.

    Parameters:
    - storage_stats (pd.DataFrame): Monthly summary of storage activity.
    """

    fig, ax1 = plt.subplots(figsize=(12, 6))
    storage_stats[["usage_cycles"]].plot(
        kind="bar",
        ax=ax1,
        width=0.4,
        position=1,
        color=["blue"],
        label="Full Cycles",
    )
    ax1.set_ylim(0, None)

    # Secondary y-axis for active hours
    ax2 = ax1.twinx()
    storage_stats[["active_hours"]].plot(
        kind="bar", ax=ax2, color=["red"], label="Active Hours", width=0.4, position=0,
    )
    ax2.set_ylim(0, None)

    # Formatting
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Cycles / Month")
    ax2.set_ylabel("Active Hours")
    ax1.set_title("Monthly Storage Cycles & Active hours (Full Year)")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Show only "YYYY-MM" format on x-axis for better readability
    ax1.set_xticklabels(storage_stats.index.strftime("%Y-%m"), rotation=45)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()


def plot_cold_output(df):
    """
    Plot the cold output per day of the CAES system with two vertical axes:
    - Left axis (ax): cold outputs
    - Right axis (ax2): cold prices
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot a horizontal line at 84 kWh
    # ax.axhline(84, color="black", linestyle="--")
    # ax.text(
    #     df.index[0],
    #     84 + 3,
    #     "Average Cold Demand per Day",
    #     color="black",
    #     fontsize=10,
    #     fontweight="bold",
    #     ha="left",
    # )

    # Plot cold outputs on the first axis
    ax.plot(
        df.index, df["cold_waste"], label="Cold Waste Output", color="grey", alpha=0.5
    )
    ax.plot(df.index, df["cold_chiller"], label="Cold Chiller Output", color="blue")
    ax.plot(df.index, q_demand_chiller, label="Cold Chiller Demand", color="purple")

    ax.plot(df.index, df["cold_freezer"], label="Cold Freezer Output", color="cyan")
    ax.plot(df.index, q_demand_freezer, label="Cold Freezer Demand", color="orange")

    # Create a second y-axis for the cold prices
    ax2 = ax.twinx()
    # ax2.plot(df.index, df["cold_price_chiller"], label="Chiller Price", color="red")
    # ax2.plot(df.index, df["cold_price_freezer"], label="Freezer Price", color="green")

    # Axis labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Cold Output [kWh]")
    # ax2.set_ylabel("Cold Price")
    ax.set_title("CAES Cold Output per Day")

    # Combine legends from both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()


def plot_results(df):
    """
    Plot all relevant results for the CAES model.
    """

    # Control the order of visualization
    plot_hourly_resolution_energy_flows_ref(start_time, end_time, df)
    plot_hourly_resolution_energy_flows(start_time, end_time, df)
    plot_cost_series(df)
    print("df columns: ", df.columns)
    # plot yearly, monthly, daily energy flows
    for resolution in ["yearly", "monthly", "daily"]:
        plot_energy_economics(
            df,
            ref_energy_columns,
            caes_energy_columns,
            resolution=resolution,
            plot_type="energy"
        )

    # plot yearly, monthly, daily cost flows
    for resolution in ["yearly", "monthly", "daily"]:
        plot_energy_economics(
            df,
            ref_cost_columns,
            caes_cost_columns,
            resolution=resolution,
            plot_type="costs"
        )

    # Plot storage usage statistics
    # Compute stats for each month
    storage_stats = calculate_monthly_storage_stats(df_l["soc"])
    # Plot full year (aggregated per month)
    plot_storage_usage_stats(storage_stats)
    plot_cold_output(df)


if __name__ == "__main__":
    dfs_to_evaluate = []

    # Preprocess results without recalculation
    df = preprocess_ref(df)
    df_original = preprocess_caes(df.copy(), results)
    dfs_to_evaluate.append(("Optimization Results", df_original))

    # Add recalculated results if recalculation is switched off (LP case)
    # if not non_simultaneity:
    #     df_recalculated = recalculate_compression_expansion(df.copy(), results)
    #     dfs_to_evaluate.append(("Recalculated (SOC-based)", df_recalculated))

    # Process and visualize all prepared DataFrames
    # for label, df_l in dfs_to_evaluate:
    df_l = df_original
    label = "Optimization Results"
    print(f"\n🔹 Evaluating: {label}")
    save_preprocessed_df(df_l)
    validate_caes_model(df_l, results)
    evaluate_economic_impact(df_l, results)
    create_energy_balance_table(df_l)
    print("peak cost ref: ", df_l["peak_cost_ref"].sum())
    create_economic_summary_table(df_l)

    # print_energy_economics(
    #     df_l, ref_energy_columns, caes_energy_columns, print_type="energy"
    #     )

    # print_energy_economics(
    #     df_l,
    #     ref_cost_columns,
    #     caes_cost_columns,
    #     print_type="costs")

    plot_results(df_l)
    # plt.show()
