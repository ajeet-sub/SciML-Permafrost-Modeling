import numpy as np

"""
Paper for referenceL
https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2019WR024969
"Projected Changes in Permafrost Active Layer Thickness Over the Qinghai‐Tibet Plateau Under Climate Change"
"""
class KudryavstevModel:
    def __init__(self, temperatures, snow_depth, veg_thickness, thermal_conductivity, heat_capacity, segment_days=30):
        self.temperatures = temperatures             # Temp List [C] (daily, monthly, or custom)
        self.snow_depth = snow_depth                 # Snow depth [m]
        self.veg_thickness = veg_thickness           # Vegetation thickness [m]
        self.k = thermal_conductivity                # Thermal conductivity [W/m·K]
        self.c = heat_capacity                       # Volumetric heat capacity [J/m^3·K]
        self.segment_days = segment_days             # Temp Data Interval in days (e.g., 30 for monthly)

    def calculate_maat(self):
        # Mean Annual Air Temperature (MAAT) temperatures
        return np.mean(self.temperatures)

    def calculate_magt(self):
        # Mean Annual Ground Temperature (MAGT) based on snow and vegetation effects
        T_a = self.calculate_maat()
        delta_Tsn = self.snow_effect()
        delta_Tveg = self.veg_effect()

        T_g = T_a + delta_Tsn + delta_Tveg
        return T_g

    def snow_effect(self):
        # Insulating effect of snow depth
        alpha_sn, beta_sn = 1.0, 0.1  # Empirical coefficients for snow effect -- may need to evaluate this
        return alpha_sn * np.exp(-beta_sn * self.snow_depth)

    def veg_effect(self):
        # Calculates the insulating effect of vegetation thickness
        alpha_veg, beta_veg = 1.0, 0.05  # Empirical coefficients for vegetation effect -- may need to evaluate this
        return alpha_veg * np.exp(-beta_veg * self.veg_thickness)

    def thawing_index(self):
        # For ALT (Stefan EQ) Thawing Index (TI) based on temperatures above 0°C and window
        TI = sum([max(0, T) * self.segment_days for T in self.temperatures])
        return TI

    def calculate_alt(self):
        # Active Layer Thickness (ALT) using Stefan eq
        TI = self.thawing_index()
        E = np.sqrt(self.k / self.c)
        ALT = E * np.sqrt(TI)
        return ALT

# Example usage
temperatures = [-10, -8, -5, 1, 10, 15, 18, 16, 8, 3, -4, -9]  # monthly temperatures in °C
segment_days = 30  # Days interval for data (adjustable for daily, weekly, monthly, etc.)
snow_depth = 0.3  # Snow depth in m
veg_thickness = 0.1  # Vegetation thickness in m
thermal_conductivity = 2.5  # W/(m·K)
heat_capacity = 2000  # J/(m^3·K)

# Initialize the model
model = KudryavstevModel(temperatures, snow_depth, veg_thickness, thermal_conductivity, heat_capacity, segment_days)

# Calculate
maat_value = model.calculate_maat()
magt_value = model.calculate_magt()
alt_value = model.calculate_alt()

print(f"Mean Annual Air Temperature (MAAT): {maat_value:.2f} °C")
print(f"Mean Annual Ground Temperature (MAGT): {magt_value:.2f} °C")
print(f"Active Layer Thickness (ALT): {alt_value:.2f} cm")
