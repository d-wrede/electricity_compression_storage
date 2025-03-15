def assign_color(col, equal_dict, mapping):
    """
    Assign a color based on the column name.
    - If the column is in equal_dict (common to both scenarios), use that color.
    - Otherwise, if the column ends with '_ref' or '_caes', remove that suffix
      and use the base name to look up the color in mapping.
    - If no mapping is found, return a default color (e.g., 'black').
    """
    if col in equal_dict:
        return equal_dict[col]
    for suffix in ["_ref", "_caes"]:
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            if base in mapping:
                return mapping[base]
    return mapping.get(col, "black")


# Define color mappings
colors = {
    "grid_import": "gray",
    "cost_grid_import": "gray",
    "peak_cost": "gray",
    "pv_feed_in": "gray",
    "pv_feed_in_earnings": "gray",
    "pv_self_use": "yellow",
    "pv_self_use_earnings": "yellow",
    "heat_earnings": "red",
    "cold_earnings": "blue",
    "compression_cost": "purple",
    "expansion_cost": "green",
    "total_cost": "black",
}

colors_equal = {
    "demand": "blue",
    "pv": "orange",
    "price": "black",
    # only for caes:
    "compression_power": "purple",
    "expansion_power": "green",
    "heat_output": "red",
    "cold_output": "blue",
}

# Example for energy flows:
ref_columns_energy = ["demand", "pv", "pv_feed_in_ref", "grid_import_ref"]
caes_columns_energy = [
    "demand",
    "pv",
    "pv_feed_in_caes",
    "grid_import_caes",
    "compression_power",
    "expansion_power",
    "heat_output",
    "cold_output",
]

# Assign colors for energy flows
# ref_colors_energy = [
#     assign_color(col, colors_equal, colors) for col in ref_columns_energy
# ]
# caes_colors_energy = [
#     assign_color(col, colors_equal, colors) for col in caes_columns_energy
# ]

# For cost flows:
ref_columns_cost = [
    "cost_grid_import_ref",
    "peak_cost_ref",
    "pv_feed_in_earnings_ref",
    "pv_self_use_earnings_ref",
]

caes_columns_cost = [
    "cost_grid_import_caes",
    "peak_cost_caes",
    "pv_feed_in_earnings_caes",
    "pv_self_use_earnings_caes",
    "heat_earnings_caes",
    "cold_earnings_caes",
    "compression_cost",
    "expansion_cost",
]

ref_columns = ref_columns_energy + ref_columns_cost
caes_columns = caes_columns_energy + caes_columns_cost

# ref_colors_cost = [assign_color(col, colors_equal, colors) for col in ref_columns_cost]
# caes_colors_cost = [
#     assign_color(col, colors_equal, colors) for col in caes_columns_cost
# ]


def assign_colors_to_columns(columns):
    """
    Returns a list of colors for the given columns based on the mapping rules.
    """
    return [assign_color(col, colors_equal, colors) for col in columns]


if __name__ == "__main__":
    pass
