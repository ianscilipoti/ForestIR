__author__ = "Colin"

import numpy as np
import os

na = np.newaxis
join = os.path.join

pi = np.pi

# Global: speed of sound in m/s (updated by set_atmosphere)
soundVel = 343.0

# --- Physical constants ---
_R_UNIV = 8.31446261815324  # J/(mol*K)
_M_DRY = 28.96546e-3        # kg/mol, typical dry air molar mass
_M_WV  = 18.01528e-3        # kg/mol, water vapour molar mass

# Specific heats at ~room temperature (mass-specific), for a lightweight mixture approximation.
_CP_DRY = 1005.0  # J/(kg*K)
_CP_WV  = 1860.0  # J/(kg*K)


def _psat_ratio_iso9613_1(T_K: float) -> float:
    """
    Saturation vapour pressure ratio Psat / Pref using a common ISO 9613-1 empirical form.
    Returns dimensionless ratio.
    """
    T01 = 273.15 + 0.01
    return 10.0 ** ((-6.8346 * (T01 / T_K) ** 1.261) + 4.6151)


def water_vapour_mole_fraction_from_rh(
    T_c: float,
    rh_percent: float,
    pressure_kPa: float = 101.325,
) -> float:
    """
    Convert relative humidity (%) to water vapour mole fraction x_v (0..1).
    """
    T_K = 273.15 + float(T_c)
    P_kPa = float(pressure_kPa)
    if P_kPa <= 0:
        raise ValueError("pressure_kPa must be > 0")

    Psat_kPa = 101.325 * _psat_ratio_iso9613_1(T_K)
    Pw_kPa = (float(rh_percent) / 100.0) * Psat_kPa

    x_v = Pw_kPa / P_kPa
    return float(np.clip(x_v, 0.0, 0.2))


def speed_of_sound_from_temp_rh_pressure(
    T_c: float,
    rh_percent: float = 50.0,
    pressure_kPa: float = 101.325,
) -> float:
    """
    Approximate speed of sound in humid air as c = sqrt(gamma * R_spec * T).
    """
    T_K = 273.15 + float(T_c)
    x_v = water_vapour_mole_fraction_from_rh(T_c, rh_percent, pressure_kPa)

    M_mix = (1.0 - x_v) * _M_DRY + x_v * _M_WV
    R_spec = _R_UNIV / M_mix

    Y_v = (x_v * _M_WV) / M_mix
    Y_d = 1.0 - Y_v

    cp = Y_d * _CP_DRY + Y_v * _CP_WV
    cv = cp - R_spec
    gamma = cp / cv

    return float(np.sqrt(gamma * R_spec * T_K))


def set_atmosphere(T_c: float, rh_percent: float = 50.0, pressure_hPa: float = 1013.25):
    """
    Set global sound speed from temperature, RH, and pressure.
    pressure_hPa is used for consistency with simulator config.
    """
    global soundVel
    soundVel = speed_of_sound_from_temp_rh_pressure(T_c, rh_percent, pressure_hPa / 10.0)
    
if __name__ == "__main__":
    for rh in [0, 50, 100]:
        print(rh, speed_of_sound_from_temp_rh_pressure(20.0, rh, 101.325))

