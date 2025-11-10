import numpy as np
kinetic_diameters = {
    'H2': 2.89,
    'He': 2.60,
    'N2': 3.64,
    'Ar': 3.40,
    'CH4': 3.80,
    'CO2': 3.30,
    'C2H4': 3.90,
    'C3H8': 4.30,
    'C2H6': 4.44,
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
    'O2': 31.998,
    'SF6': 146.05
}
T = 22 + 273.15  # 22 C in Kelvin
R =  8.31446261815324 # J/(mol·K)
N_A = 6.02214076e23  # molecules per mole

molar_masses_kg_per_mol = {gas: mw / 1000 for gas, mw in molecular_weights.items()}  # kg/mol