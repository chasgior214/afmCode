import numpy as np
kinetic_diameters_Breck = {
    'H2': 2.89,
    'He': 2.60,
    'N2': 3.64,
    'Ar': 3.40,
    'CH4': 3.80,
    'CO2': 3.30,
    'C2H4': 3.90,
    'C3H8': 4.30,
    'C2H6': 4.44, # Breck did not give a value for C2H6
    'n-C4H10': 4.3,
    'O2': 3.46,
    'SF6': 5.50
}
molecular_weights = {
    'H2': 2.016,
    'He': 4.0026,
    'N2': 28.014,
    'Ar': 39.95,
    'CH4': 16.043,
    'CO2': 44.009,
    'C2H4': 28.054,
    'C3H8': 44.097,
    'C2H6': 30.070,
    'n-C4H10': 58.124,
    'O2': 31.998,
    'SF6': 146.05
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
    'H2': None,
    'He': None,
    'N2': 2.991, # [1]
    'Ar': 3.51, # [2]
    'CH4': 3.829, # [1]
    'CO2': 3.189, # [1]
    'C2H4': 3.28, # [3]
    'C3H8': 4.020, # [1]
    'C2H6': 3.809, # [1]
    'n-C4H10': 4.01, # [2]
    'O2': 2.930, # [1]
    'SF6': 4.871 # [1]
}
min_2_dimensions = {
    'H2': None,
    'He': None,
    'N2': 3.054, # [1]
    'Ar': 3.63, # [2]
    'CH4': 3.942, # [1]
    'CO2': 3.339, # [1]
    'C2H4': 4.18, # [3]
    'C3H8': 4.516, # [1]
    'C2H6': 4.079, # [1]
    'n-C4H10': 4.52, # [2]
    'O2': 2.985, # [1]
    'SF6': 5.266 # [1]
}


# https://ntrs.nasa.gov/citations/19630012982
Svehla_LJ_diameters = {
    'H2': 2.827,
    'He': 2.551,
    'N2': 3.798,
    'Ar': 3.542,
    'CH4': 3.758,
    'CO2': 3.941,
    'C2H4': 4.163,
    'C3H8': 5.118,
    'C2H6': 4.443,
    'n-C4H10': 4.687,
    'O2': 3.467,
    'SF6': 5.128
}

if __name__ == "__main__":
    # plot the kinetic diameters from all sources for comparison
    import matplotlib.pyplot as plt
    gases = ['H2', 'He', 'O2', 'N2', 'Ar', 'CO2', 'CH4', 'C2H4', 'C2H6', 'C3H8']
    indices = np.arange(len(gases))
    width = 0.15

    breck_values = [kinetic_diameters_Breck[gas] for gas in gases]
    min1_values = [min_1_dimensions[gas] if min_1_dimensions[gas] is not None else 0 for gas in gases]
    min2_values = [min_2_dimensions[gas] if min_2_dimensions[gas] is not None else 0 for gas in gases]
    svehla_values = [Svehla_LJ_diameters[gas] for gas in gases]

    plt.bar(indices - 1.5*width, breck_values, width, label='Breck')
    plt.bar(indices - 0.5*width, svehla_values, width, label='Svehla LJ σ')
    plt.bar(indices + 0.5*width, min1_values, width, label='MIN-1')
    plt.bar(indices + 1.5*width, min2_values, width, label='MIN-2')

    plt.xticks(indices, gases)
    plt.ylabel('Kinetic Diameter (Å)')
    plt.title('Comparison of Kinetic Diameters from Different Sources')
    plt.legend()
    plt.show()