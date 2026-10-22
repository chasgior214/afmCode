import numpy as np

import gas_constants as gc # will use R, T, P_ATM, molar_masses_kg_per_mol

OXIDE_DEPTH = 285e-9
DRIE_ETCH_DEPTH = 2.1e-6
WELL_DIAM = 3e-6 # DRIE etched diameter. Wells will have a larger surface diameter due to undercut of the photoresist during HF etching of the oxide. The rest of the well is etched isotropically by DRIE with the photoresist on and should be ~3um.

DRIE_VOLUME_BELOW_SUBSTRATE = DRIE_ETCH_DEPTH * np.pi * (WELL_DIAM / 2) ** 2

# TODO include all equations from our paper that I may ever use, and in the order of the script show how they feed into each other
# TODO also include the mass-normalization and use the function for it here in plot_permeation.py

# needed: slope, initial height
    # compare using slope(h) and h vs slope(h_initial) and h_initial vs h at the middle of the height range of points used to get slope, but should be about the same

# Multiplying by sqrt(2 pi M R T) will give effective area

""" Knudsen effusion model for permeation of gas through a small orifice in a thin barrier

permeation coefficient:
q = A / sqrt(2 pi M R T)
"""

def knudsen_effusion_predicted_area(q, M, T=gc.T):
    """Calculate the area of a small orifice in a thin barrier predicted by Knudsen effusion

    Parameters
    ----------
    q : float
        Permeation coefficient (mol/s/Pa)
    M : float
        Molar mass of the gas (kg/mol)
    T : float
        Temperature (K)
    """
    return q * np.sqrt(2 * np.pi * M * gc.R * T)


""" Hencky's solution for the deflection of an elastic, circular membrane exposed to
a uniform pressure diference (https://academic.oup.com/qjmam/article/9/1/84/1832782)
predicts that the internal pressure varies with the maximum membrane deflection (h) as
P + (P_0 - P_atm) = c h^3
where c is a material- and membrane thickness-dependent constant
    - c for two membranes under the same graphene sheet is nearly the same
"""


def hencky_pressure(h, c, P_0, P_atm=gc.P_ATM):
    """Calculate the pressure of the gas of interest in a cavity using Hencky's solution

    Parameters
    ----------
    h : float
        Height of the membrane deflection (m)
    c : float
        Material- and membrane thickness-dependent constant (kPa/m^3)
    P_0 : float
        Pressure of air partial pressure in the cavity (kPa)
    P_atm : float
        Atmospheric pressure (kPa)
    """
    return c * h ** 3 + P_atm - P_0

def hencky_height(P, c, P_0, P_atm=gc.P_ATM):
    """Calculate the height of a deflected membrane using Hencky's solution

    Parameters
    ----------
    P : float
        Pressure in the cavity (kPa)
    c : float
        Material- and membrane thickness-dependent constant (kPa/m^3)
    P_0 : float
        Pressure of air partial pressure in the cavity (kPa)
    P_atm : float
        Atmospheric pressure (kPa)
    """
    return ((P - P_atm + P_0) / c) ** (1 / 3)

def perm_coeff(h, dhdt, P_charge, P_air_cav, well_depth=DRIE_ETCH_DEPTH, well_diam=WELL_DIAM, a=0.5):
    """Calculate the permeation coefficient of a membrane

    Parameters
    ----------
    h : float
        Height of the membrane deflection (m)
    dhdt : float
        Rate of change of the membrane deflection in the linear region (m/s)
    P_charge : float
        Pressure of gas of interest in the charged cavity (kPa)
    P_air_cav : float
        Pressure of air in the cavity (kPa)
    well_depth : float
        Depth of the well (m)
    well_diam : float
        Diameter of the well (m)
    a : float
        Geometric factor giving the ratio of the volume of the bulged membrane above the nominal top of the cavity to the volume of a cylinder with the radius of the cavity and height of the bulge
    """
    well_area = np.pi * (well_diam / 2) ** 2
    return dhdt * (3 * well_depth / h + 4 * a - (3 * well_depth / h + 3 * a) * (gc.P_ATM - P_air_cav) / P_charge) * well_area / (gc.R * gc.T)

# Paper values for comparison:
# He	    H2	        O2	        Ar	        CH4	        N2	        CO2	        C2H4	    C2H6	    C3H8
# 3.78E-23	2.46E-23	2.84E-24	1.05E-24	1.40E-24	1.50E-24	3.74E-23	3.61E-24	7.36E-26	3.22E-26


print(perm_coeff(179.35e-9, 20.648e-9 / 60, 298, 0))
print(perm_coeff(169.25e-9, 29.5e-9 / 60, 310, 0))
print(perm_coeff(197.83e-9, 1.35e-9 / 60, 315, 0))
