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

molar_masses = {gas: mw / 1000 for gas, mw in molecular_weights.items()}  # kg/mol

if __name__ == "__main__":
    H2_slopes = {
        'green': 2.1,
        'red': 44.4,
        'blue': 30.3,
        'orange': 2.2,
        'black': 2.3
    }
    He_slopes = {
        'green': 9,
        'red': 25.1,
        'blue': 39.7,
        'orange': 8.8,
        'black': 8.6
    }
    CO2_slopes = {
        'green': 0.86,
        'orange': 0.67,
        'black': 0.83
    }
    Ar_slopes = {
        'red': 0.95,
        'blue': 2.7,
        'green': 0.088,
        'orange': 0.059,
        'black': 0.080
    }
    CH4_slopes = {
        'red': 0.4,
        'blue': 0.2
    }
    N2_slopes = {
        'green': 0.026,
        'red': 0.72,
        'blue': 1.51,
        'orange': 0.019,
        'black': 0.017
    }
    color_to_marker = {
    'black': 'x',
    'green': '+',
    'orange': 'o',
    'blue': '*',
    'red': '^'
    }
    unfilled_colors = {'orange', 'red'}
    # plot the slopes versus kinetic diameters. Colour the points by the keys of the slope dictionaries. Make the orange and red point markers unfilled.
    import matplotlib.pyplot as plt
    for gas, slopes in [('H2', H2_slopes), ('He', He_slopes), ('CO2', CO2_slopes), ('Ar', Ar_slopes), ('CH4', CH4_slopes), ('N2', N2_slopes)]:
        kd = kinetic_diameters[gas]
        for color, slope in slopes.items():
            marker = color_to_marker[color]
            if color in unfilled_colors:
                plt.scatter(kd, slope, label=f"{gas} - {color}", edgecolors=color, facecolors='none', marker=marker, s=100)
            else:
                plt.scatter(kd, slope, label=f"{gas} - {color}", color=color, marker=marker, s=100)
    plt.yscale('log')

    # put labels for each gas kinematic diameters on the x axis
    for gas, slopes in [('H2', H2_slopes), ('He', He_slopes), ('CO2', CO2_slopes), ('Ar', Ar_slopes), ('CH4', CH4_slopes), ('N2', N2_slopes)]:
        kd = kinetic_diameters[gas]
        plt.axvline(x=kd, color='gray', linestyle='--', linewidth=0.5)
        plt.text(kd, plt.ylim()[0], gas, verticalalignment='bottom', horizontalalignment='right')
    plt.xlabel('Kinetic Diameter (Å)')
    plt.ylabel('Deflation Curve Slope (nm/min)')
    plt.title('Deflation Curve Slopes vs Gas Kinetic Diameter')
    plt.grid(True)
    plt.show()


    # do the same but plot slopes versus molecular weights
    plt.figure()
    for gas, slopes in [('H2', H2_slopes), ('He', He_slopes), ('CO2', CO2_slopes), ('Ar', Ar_slopes), ('CH4', CH4_slopes), ('N2', N2_slopes)]:
        mw = molecular_weights[gas]
        for color, slope in slopes.items():
            marker = color_to_marker[color]
            if color in unfilled_colors:
                plt.scatter(mw, slope, label=f"{gas} - {color}", edgecolors=color, facecolors='none', marker=marker, s=100)
            else:
                plt.scatter(mw, slope, label=f"{gas} - {color}", color=color, marker=marker, s=100)
    plt.yscale('log')
    # put labels for each gas molecular weights on the x axis
    for gas, slopes in [('H2', H2_slopes), ('He', He_slopes), ('CO2', CO2_slopes), ('Ar', Ar_slopes), ('CH4', CH4_slopes), ('N2', N2_slopes)]:
        mw = molecular_weights[gas]
        plt.axvline(x=mw, color='gray', linestyle='--', linewidth=0.5)
        plt.text(mw, plt.ylim()[0], gas, verticalalignment='bottom', horizontalalignment='right')
    plt.xlabel('Molecular Weight (amu)')
    plt.xlim(left=0)
    plt.ylabel('Deflation Curve Slope (nm/min)')
    plt.title('Deflation Curve Slopes vs Gas Molecular Weight')
    plt.grid(True)
    plt.show()

    # make a plot with diameter on the x axis and slope normalized by multiplying by the square root of (molar mass * 2 * π * R * T) on the y axis
    plt.figure()
    for gas, slopes in [('H2', H2_slopes), ('He', He_slopes), ('CO2', CO2_slopes), ('Ar', Ar_slopes), ('CH4', CH4_slopes), ('N2', N2_slopes)]:
        kd = kinetic_diameters[gas]
        mm = molar_masses[gas]
        for color, slope in slopes.items():
            norm_slope = slope * ((mm * 2 * np.pi * R * T) ** 0.5)
            marker = color_to_marker[color]
            if color in unfilled_colors:
                plt.scatter(kd, norm_slope, label=f"{gas} - {color}", edgecolors=color, facecolors='none', marker=marker, s=100)
            else:
                plt.scatter(kd, norm_slope, label=f"{gas} - {color}", color=color, marker=marker, s=100)
    plt.yscale('log')
    # put labels for each gas kinematic diameters on the x axis
    for gas, slopes in [('H2', H2_slopes), ('He', He_slopes), ('CO2', CO2_slopes), ('Ar', Ar_slopes), ('CH4', CH4_slopes), ('N2', N2_slopes)]:
        kd = kinetic_diameters[gas]
        plt.axvline(x=kd, color='gray', linestyle='--', linewidth=0.5)
        plt.text(kd, plt.ylim()[0], gas, verticalalignment='bottom', horizontalalignment='right')
    plt.xlabel('Kinetic Diameter (Å)')
    plt.ylabel('Normalized Deflation Curve Slope (nm/min * sqrt(kg * J)/mol)')
    plt.title('Normalized Deflation Curve Slopes vs Gas Kinetic Diameter')
    plt.grid(True)
    plt.show()