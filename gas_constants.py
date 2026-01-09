import numpy as np
kinetic_diameters_Breck = {
    'H2': 2.89,
    'He': 2.60,
    'Ne': 2.75,
    'Ar': 3.40,
    'Kr': 3.60,
    'Xe': 3.96,

    'N2': 3.64,
    'O2': 3.46,
    'CO': 3.76,
    'CO2': 3.30,
    'SF6': 5.50,

    'CH4': 3.80,
    'C2H2': 3.30,
    'C2H4': 3.90,
    'C2H6': 4.44, # Breck did not give a value for C2H6, but this value is widely used alongside his values
    'C3H8': 4.30,
    'n-C4H10': 4.3
}

molecular_weights = {
    'H2': 2.016,
    'He': 4.0026,
    'Ne': 20.180,
    'Ar': 39.95,
    'Kr': 83.798,
    'Xe': 131.293,

    'N2': 28.014,
    'O2': 31.998,
    'CO': 28.010,
    'CO2': 44.009,
    'SF6': 146.05,

    'CH4': 16.043,
    'C2H2': 26.038,
    'C2H4': 28.054,
    'C2H6': 30.070,
    'C3H8': 44.097,
    'n-C4H10': 58.124,
    'CH3OCH3': 46.069
}

T = 22 + 273.15  # 22 C in Kelvin
R =  8.31446261815324 # J/(mol·K)
N_A = 6.02214076e23  # molecules per mole

molar_masses_kg_per_mol = {gas: mw / 1000 for gas, mw in molecular_weights.items()}  # kg/mol

# MIN-1 and MIN-2 sources:
# [1] https://pubs.acs.org/doi/10.1021/ja973906m
# [2] https://pubs.acs.org/doi/10.1021/ja984122r
# [3] https://pubs.acs.org/doi/10.1021/jp0108263

min_1_dimensions = {
    'N2': 2.991, # [1]
    'Ar': 3.51, # [2]
    'Xe': 4.04, # [2]
    'CH4': 3.829, # [1]
    'CO': 3.280, # [1]
    'CO2': 3.189, # [1]
    'C2H4': 3.28, # [3]
    'C3H8': 4.020, # [1]
    'C2H6': 3.809, # [1]
    'n-C4H10': 4.01, # [2]
    'O2': 2.930, # [1]
    'SF6': 4.871, # [1]
    'CH3OCH3': 4.083 # [1]
}
min_2_dimensions = {
    'N2': 3.054, # [1]
    'Ar': 3.63, # [2]
    'Xe': 4.17, # [2]
    'CH4': 3.942, # [1]
    'CO': 3.339, # [1]
    'CO2': 3.339, # [1]
    'C2H4': 4.18, # [3]
    'C3H8': 4.516, # [1]
    'C2H6': 4.079, # [1]
    'n-C4H10': 4.52, # [2]
    'O2': 2.985, # [1]
    'SF6': 5.266, # [1]
    'CH3OCH3': 4.127 # [1]
}


# https://ntrs.nasa.gov/citations/19630012982
    # method 1 is least-squares fit of experimental viscosity data
    # method 2 is graphically determined using experimental viscosity data
Svehla_LJ_diameters = {
    'H2': 2.827, # method 1
    'He': 2.551, # method 2
    'Ne': 2.820, # method 1
    'Ar': 3.542, # method 1
    'Kr': 3.655, # method 1
    'Xe': 4.047, # method 1

    'N2': 3.798, # method 1
    'O2': 3.467, # method 1
    'CO': 3.690, # method 1
    'CO2': 3.941, # method 1
    'SF6': 5.128, # method 1

    'CH4': 3.758, # method 1
    'C2H2': 4.033, # method 1
    'C2H4': 4.163, # method 1
    'C2H6': 4.443, # method 1
    'C3H8': 5.118, # method 1
    'n-C4H10': 4.687, # method 1
    'CH3OCH3': 4.307 # method 1
}

if __name__ == "__main__":
    # plot the kinetic diameters from all sources for comparison
    import matplotlib.pyplot as plt
    gases = [gas for gas in Svehla_LJ_diameters.keys()]
    indices = np.arange(len(gases))
    width = 0.15

    min1_values = [min_1_dimensions[gas] if gas in min_1_dimensions else 0 for gas in gases]
    min2_values = [min_2_dimensions[gas] if gas in min_2_dimensions else 0 for gas in gases]
    Breck_values = [kinetic_diameters_Breck[gas] if gas in kinetic_diameters_Breck else 0 for gas in gases]

    plt.bar(indices - 1.5*width, Breck_values, width, label='Breck')
    plt.bar(indices - 0.5*width, [Svehla_LJ_diameters[gas] for gas in gases], width, label='Svehla LJ σ')
    plt.bar(indices + 0.5*width, min1_values, width, label='MIN-1')
    plt.bar(indices + 1.5*width, min2_values, width, label='MIN-2')

    plt.xticks(indices, gases)
    plt.ylabel('Kinetic Diameter (Å)')
    plt.title('Comparison of Kinetic Diameters from Different Sources')
    plt.legend()
    plt.show()

    # plot molecular weights vs Svehla LJ diameters and Breck diameters
    plt.figure()
    svehla_mw = [molecular_weights[gas] for gas in gases if gas in Svehla_LJ_diameters]
    svehla_diameters = [Svehla_LJ_diameters[gas] for gas in gases if gas in Svehla_LJ_diameters]
    Breck_mw = [molecular_weights[gas] for gas in gases if gas in kinetic_diameters_Breck]
    Breck_diameters = [kinetic_diameters_Breck[gas] for gas in gases if gas in kinetic_diameters_Breck]
    plt.scatter(svehla_mw, svehla_diameters, label='Svehla LJ σ', color='blue', marker='x')
    plt.scatter(Breck_mw, Breck_diameters, label='Breck', color='orange', marker='+')
    ax = plt.gca()
    # Sort gases by molecular weight to handle overlapping labels
    sorted_gases = sorted(gases, key=lambda g: molecular_weights.get(g, 0))
    prev_mw = -999
    min_gap = 1  # minimum gap in MW units before labels overlap
    y_offsets = [-0.02, -0.07, -0.12]  # staggered y positions
    offset_idx = 0
    for gas in sorted_gases:
        mw = molecular_weights.get(gas)
        if mw is None:
            continue
        if mw - prev_mw < min_gap:
            offset_idx = (offset_idx + 1) % len(y_offsets)
        else:
            offset_idx = 0
        ax.axvline(mw, color='gray', alpha=0.3, linewidth=0.7)
        ax.text(
            mw,
            y_offsets[offset_idx],
            gas,
            rotation=90,
            va='top',
            ha='center',
            transform=ax.get_xaxis_transform(),
            fontsize=8,
        )
        prev_mw = mw
    plt.subplots_adjust(bottom=0.25)
    plt.xlim(left=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.subplots_adjust(bottom=0.15)
    plt.legend(loc='upper left')
    plt.xlabel('Molecular Weight (g/mol)')
    plt.ylabel('Molecular Size (Å)')
    plt.show()