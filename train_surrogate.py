import math
import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CHEM_DT = 10.0
STATE_DIM = 4
INPUT_DIM = 14


class AtmosphereType(Enum):
    """Different atmospheric scenarios for training diversity."""
    URBAN_HIGH_TRAFFIC = "urban_high_traffic"
    INDUSTRIAL = "industrial"
    SUBURBAN = "suburban"
    MORNING_RUSH = "morning_rush"
    EVENING_RUSH = "evening_rush"
    CLEAN_DAY = "clean_day"
    AFTERNOON_HAZE = "afternoon_haze"


@dataclass(frozen=True)
class TrainConfig:
    samples: int = int(os.getenv("SMOG_TRAIN_SAMPLES", "45000"))
    epochs: int = int(os.getenv("SMOG_EPOCHS", "32"))
    batch_size: int = int(os.getenv("SMOG_BATCH_SIZE", "512"))
    learning_rate: float = float(os.getenv("SMOG_LR", "0.001"))
    seed: int = int(os.getenv("SMOG_SEED", "42"))
    use_multiple_atmospheres: bool = True
    show_status: bool = True


CFG = TrainConfig()
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

try:
    import onnx  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'onnx'. Install it in the same venv with `pip install onnx` before training."
    ) from exc

# Chemical kinetic constants (redesigned for realistic NO₂ dynamics)
K_O3_NO = 2.5e-2           # O₃ + NO → NO₂ + O₂ reaction rate
K_RO2_NO = 2.2e-3          # VOC radical + NO → NO₂ pathway
K_VOC_LOSS = 6.0e-5        # VOC atmospheric oxidation loss
J1_MAX = 2.0e-2            # ENHANCED: NO₂ photolysis rate (increased from 7.5e-3)
K_O3_DEP = 2.0e-4          # O₃ dry deposition to surface
K_NO2_DEP = 2.5e-4         # ENHANCED: NO₂ deposition (increased from 7.0e-5 to match O₃)
K_NO2_HNO3 = 1.5e-4        # NEW: NO₂ oxidation to HNO₃ loss pathway (wet deposition)
K_MIX_BASE = 6.0e-5        # Base atmosphere mixing rate
K_MIX_DAY = 2.2e-4         # Daytime thermal mixing
K_MIX_WIND = 2.4e-4        # Wind-driven mixing
K_O3_HUMIDITY_LOSS = 8.5e-5 # Humidity-enhanced O₃ loss
K_TRAP_RELEASE = 1.2e-4    # Inversion release rate

# Realistic state limits based on EPA/atmospheric standards
STATE_MAX = np.array([90.0, 120.0, 140.0, 600.0], dtype=np.float64)   # NO₂, NO, O₃, VOC (ppb)
STATE_MIN = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def calculate_realistic_smog_index(no2: float, o3: float, voc: float, humidity: float) -> float:
    """
    Calculate a more realistic Air Quality Index (AQI-like) smog level.
    
    The smog intensity is primarily driven by:
    - O₃ (ground-level ozone): Primary pollutant, highly toxic
    - NO₂ (nitrogen dioxide): Secondary contributor, respiratory irritant
    - VOC (volatile organics): Precursor, indicates potential continued formation
    - Humidity: Affects particle formation and persistence
    
    Returns value in [0, 1] where >0.5 is "unhealthy" air quality.
    """
    # Normalize individual components against EPA/WHO standards
    # These are realistic thresholds for concern
    
    # O₃: Main air quality determinant
    # EPA: 70 ppb 8-hr standard; 150+ ppb is hazardous
    o3_normalized = min(1.0, o3 / 150.0)  # 150 ppb = most severe
    
    # NO₂: Respiratory irritant
    # EPA: 200 ppb 1-hr standard
    no2_normalized = min(1.0, no2 / 200.0)
    
    # VOC: Indicates photochemical potential (longer-term concern)
    # Higher VOC = more likely to form ozone over next few hours
    voc_normalized = min(1.0, voc / 500.0)
    
    # Humidity factor: High humidity + high ozone = worse health effects
    # Morning fog or high humidity can concentrate pollutants
    humidity_amplify = 0.8 + 0.4 * max(0.0, humidity - 0.5)  # Range [0.8, 1.2]
    
    # Weighted combination: O₃ dominates, NO₂ and VOC contribute
    # This reflects real-world health impacts
    smog_level = (
        0.60 * o3_normalized * humidity_amplify +    # Primary: ozone (60%), amplified by humidity)
        0.25 * no2_normalized +                        # Secondary: NO₂ (25%)
        0.15 * voc_normalized                          # Tertiary: VOC persistence (15%)
    )
    
    return float(np.clip(smog_level, 0.0, 1.0))


def gaussian_wrap(hour: float, center: float, width: float) -> float:
    raw = abs(hour - center)
    distance = min(raw, 24.0 - raw)
    return math.exp(-0.5 * (distance / width) ** 2)


def solar_arc(hour: float) -> float:
    if hour < 6.0 or hour > 18.0:
        return 0.0
    return max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))


def hour_features(hour: float) -> tuple[float, float]:
    theta = 2.0 * math.pi * (hour % 24.0) / 24.0
    return math.sin(theta), math.cos(theta)


def j1(hour: float, solar_flux: float) -> float:
    return J1_MAX * (solar_arc(hour) ** 1.25) * solar_flux


def traffic_time_factor(hour: float, weekend_mode: float) -> float:
    morning = gaussian_wrap(hour, 8.0, 1.45 if weekend_mode >= 0.5 else 1.1)
    evening = gaussian_wrap(hour, 17.5, 1.7 if weekend_mode >= 0.5 else 1.25)
    midday = gaussian_wrap(hour, 12.8, 2.5)
    midnight = gaussian_wrap(hour, 0.2, 2.3)

    weekday_profile = (0.12 + 0.76 * morning + 0.72 * evening + 0.36 * midday) * (1.0 - 0.82 * midnight)
    weekend_profile = (0.10 + 0.42 * morning + 0.52 * evening + 0.34 * midday) * (1.0 - 0.78 * midnight)
    profile = weekend_profile if weekend_mode >= 0.5 else weekday_profile
    return float(np.clip(profile, 0.05, 1.0))


def effective_traffic_density(hour: float, traffic_density: float, weekend_mode: float) -> float:
    return traffic_density * traffic_time_factor(hour, weekend_mode)


def traffic_profile(hour: float, traffic_density: float, weekend_mode: float) -> float:
    density_now = effective_traffic_density(hour, traffic_density, weekend_mode)
    weekend_scale = 0.88 if weekend_mode >= 0.5 else 1.0
    return weekend_scale * (0.08 + 1.70 * density_now)


def mixing_coeff(hour: float, solar_flux: float, wind_speed: float, inversion_strength: float) -> float:
    midday_phase = min(1.0, max(0.0, (hour - 8.5) / 8.5))
    mixed_day = max(0.0, math.sin(math.pi * midday_phase))
    inversion_drag = max(0.18, 1.0 - 0.72 * inversion_strength)
    return (K_MIX_BASE + K_MIX_DAY * mixed_day * solar_flux + K_MIX_WIND * wind_speed) * inversion_drag


def trapping_factor(inversion_strength: float, wind_speed: float) -> float:
    return min(2.1, max(0.55, 1.0 + 1.2 * inversion_strength - 0.55 * wind_speed))


def temperature_factor(temp_c: float) -> float:
    return min(1.45, max(0.65, 0.82 + 0.018 * (temp_c - 18.0)))


def humidity_factor(humidity: float) -> float:
    return min(1.22, max(0.75, 0.82 + 0.36 * humidity))


def background_state(
    hour: float,
    traffic_density: float,
    solar_flux: float,
    wind_speed: float,
    temp_c: float,
    industrial: float,
    weekend_mode: float,
) -> np.ndarray:
    """Background state: remote or well-mixed reference levels."""
    traffic_now = effective_traffic_density(hour, traffic_density, weekend_mode)
    weekend_bias = 0.85 if weekend_mode >= 0.5 else 1.0
    solar = solar_arc(hour)
    
    # NO₂ background: low daytime (photolyzed), higher night (accumulates)
    no2_bg = 1.5 + 1.2 * gaussian_wrap(hour, 7.0, 2.0) + 1.5 * industrial * weekend_bias
    no2_bg *= (0.3 + 0.7 * (1.0 - solar))  # Reduced by 70% during daytime
    no2_bg = np.clip(no2_bg, 0.3, 12.0)  # REDUCED: max 12 ppb (was 35)
    
    # NO background: very low, mostly consumed by O₃
    no_bg = np.clip(0.1 + 0.2 * gaussian_wrap(hour, 8.0, 1.0), 0.0, 1.0)
    
    # O₃ background: higher during day (photochemical production), lower at night
    o3_bg = np.clip(15.0 + 12.0 * solar * solar_flux + 8.0 * wind_speed, 8.0, 45.0)  # REDUCED: max 45 ppb (was 70)
    
    # VOC background: varies with traffic, lower during day (oxidation), higher at night
    voc_bg = np.clip(20.0 + 8.0 * traffic_now + 12.0 * industrial, 10.0, 90.0)  # REDUCED: max 90 ppb (was 130)
    
    return np.array([no2_bg, no_bg, o3_bg, voc_bg], dtype=np.float64)


def urban_baseline(
    hour: float,
    traffic_density: float,
    solar_flux: float,
    wind_speed: float,
    temp_c: float,
    humidity: float,
    industrial: float,
    inversion: float,
    weekend_mode: float,
) -> np.ndarray:
    """Urban baseline: typical concentration levels for given conditions."""
    rush = traffic_profile(hour, traffic_density, weekend_mode)
    day = solar_arc(hour) * solar_flux
    trap = trapping_factor(inversion, wind_speed)
    solar = solar_arc(hour)
    
    # NO₂ urban baseline: peaks during morning/evening traffic, depleted by photolysis during day
    # Realistic EPA-observed: 20-40 ppb typical urban, 50+ ppb only in severe pollution
    no2_urban = 8.0 + 12.0 * rush + 6.0 * industrial * trap
    no2_urban *= (0.2 + 0.8 * (1.0 - solar * solar_flux))  # Reduced by photolysis during day
    no2_urban = np.clip(no2_urban, 2.0, 50.0)  # REALISTIC: max 50 ppb for urban (was 150)
    
    # NO urban baseline: produced by photolysis, quickly consumed by O₃
    no_urban = np.clip(5.0 + 8.0 * rush * trap * (1.0 - solar), 0.5, 30.0)  # REDUCED max
    
    # O₃ urban baseline: increases during afternoon photochemistry, low at night
    o3_urban = np.clip(8.0 + 18.0 * day + 6.0 * wind_speed + 4.0 * industrial, 3.0, 90.0)
    
    # VOC urban baseline: from traffic and industrial emissions
    voc_urban = np.clip(
        45.0 + 70.0 * rush + 60.0 * industrial + 3.0 * humidity + 0.6 * max(0.0, temp_c - 20.0),
        20.0,
        500.0,  # REDUCED: max 500 ppb
    )
    
    return np.array([no2_urban, no_urban, o3_urban, voc_urban], dtype=np.float64)


def clamp_state(y: np.ndarray) -> np.ndarray:
    """Clamp chemical state to realistic bounds to prevent unrealistic values."""
    return np.minimum(np.maximum(y, STATE_MIN), STATE_MAX)


def sample_atmospheric_params(atmosphere_type: AtmosphereType) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Generate realistic atmospheric parameters for different scenarios.
    Returns: hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode
    """
    if atmosphere_type == AtmosphereType.URBAN_HIGH_TRAFFIC:
        # Dense urban area with heavy traffic
        hour = np.random.uniform(6.0, 20.0)  # Peak pollution hours
        traffic_density = np.random.uniform(0.6, 1.0)  # Heavy traffic
        solar_flux = np.random.uniform(0.5, 1.3)
        wind_speed = np.random.uniform(0.0, 0.5)  # Stagnant conditions trap pollution
        temp_c = np.random.uniform(15.0, 32.0)
        humidity = np.random.uniform(0.3, 0.8)
        industrial = np.random.uniform(0.1, 0.4)
        inversion = np.random.uniform(0.2, 0.8)  # Thermal inversion traps pollution
        weekend_mode = 0.0  # Weekday profile
        
    elif atmosphere_type == AtmosphereType.INDUSTRIAL:
        # Industrial zone with heavy emissions
        hour = np.random.uniform(6.0, 18.0)
        traffic_density = np.random.uniform(0.3, 0.7)
        solar_flux = np.random.uniform(0.4, 1.2)
        wind_speed = np.random.uniform(0.0, 0.6)
        temp_c = np.random.uniform(12.0, 28.0)
        humidity = np.random.uniform(0.2, 0.7)
        industrial = np.random.uniform(0.7, 1.0)  # Heavy industrial emissions
        inversion = np.random.uniform(0.3, 0.9)
        weekend_mode = 0.0
        
    elif atmosphere_type == AtmosphereType.SUBURBAN:
        # Suburban area: moderate pollution
        hour = np.random.uniform(7.0, 19.0)
        traffic_density = np.random.uniform(0.3, 0.6)
        solar_flux = np.random.uniform(0.6, 1.4)
        wind_speed = np.random.uniform(0.3, 1.0)  # Better ventilation
        temp_c = np.random.uniform(14.0, 30.0)
        humidity = np.random.uniform(0.4, 0.8)
        industrial = np.random.uniform(0.1, 0.3)  # Light industrial
        inversion = np.random.uniform(0.1, 0.5)  # Weaker inversions
        weekend_mode = np.random.choice([0.0, 1.0])
        
    elif atmosphere_type == AtmosphereType.MORNING_RUSH:
        # Rush hour morning: traffic spike
        hour = np.random.uniform(7.0, 9.5)
        traffic_density = np.random.uniform(0.7, 1.0)  # Peak traffic
        solar_flux = np.random.uniform(0.2, 0.8)  # Low sun angle
        wind_speed = np.random.uniform(0.0, 0.4)
        temp_c = np.random.uniform(10.0, 20.0)
        humidity = np.random.uniform(0.5, 0.9)  # Morning dew/fog common
        industrial = np.random.uniform(0.0, 0.3)
        inversion = np.random.uniform(0.2, 0.7)  # Morning inversions
        weekend_mode = np.random.choice([0.0, 1.0])
        
    elif atmosphere_type == AtmosphereType.EVENING_RUSH:
        # Evening rush: second traffic spike
        hour = np.random.uniform(16.5, 18.5)
        traffic_density = np.random.uniform(0.6, 1.0)
        solar_flux = np.random.uniform(0.3, 1.0)
        wind_speed = np.random.uniform(0.1, 0.6)
        temp_c = np.random.uniform(18.0, 32.0)
        humidity = np.random.uniform(0.3, 0.7)
        industrial = np.random.uniform(0.0, 0.3)
        inversion = np.random.uniform(0.3, 0.8)  # Evening inversions common
        weekend_mode = np.random.choice([0.0, 1.0])
        
    elif atmosphere_type == AtmosphereType.CLEAN_DAY:
        # Clean/rural day: good ventilation, low emissions
        hour = np.random.uniform(8.0, 16.0)  # Daytime
        traffic_density = np.random.uniform(0.0, 0.3)  # Light traffic
        solar_flux = np.random.uniform(0.8, 1.5)
        wind_speed = np.random.uniform(0.5, 1.0)  # Good wind conditions
        temp_c = np.random.uniform(16.0, 28.0)
        humidity = np.random.uniform(0.3, 0.6)
        industrial = np.random.uniform(0.0, 0.1)  # Minimal industrial
        inversion = np.random.uniform(0.0, 0.3)  # Weak/no inversion
        weekend_mode = 1.0  # Often weekend
        
    else:  # AFTERNOON_HAZE
        # Afternoon photochemical smog peak
        hour = np.random.uniform(13.0, 16.0)  # Peak sun + peak precursors
        traffic_density = np.random.uniform(0.4, 0.8)
        solar_flux = np.random.uniform(1.0, 1.5)  # Strong sun
        wind_speed = np.random.uniform(0.0, 0.5)  # Stagnant
        temp_c = np.random.uniform(24.0, 35.0)  # Hot = higher reaction rates
        humidity = np.random.uniform(0.2, 0.6)
        industrial = np.random.uniform(0.1, 0.5)
        inversion = np.random.uniform(0.4, 0.9)  # Strong inversion
        weekend_mode = 0.0
    
    return hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode


def derivatives(
    y: np.ndarray,
    hour: float,
    traffic_density: float,
    solar_flux: float,
    wind_speed: float,
    temp_c: float,
    humidity: float,
    industrial: float,
    inversion: float,
    weekend_mode: float,
) -> np.ndarray:
    """Compute chemical time derivatives for NO₂, NO, O₃, VOC.
    
    REDESIGNED for realistic NO₂ dynamics:
    - Stronger photolysis (J1_MAX: 7.5e-3 → 2.0e-2, +2.7x)
    - Stronger NO₂ deposition (K_NO2_DEP: 7.0e-5 → 2.5e-4, +3.6x)
    - New NO₂ oxidation pathway (K_NO2_HNO3: 1.5e-4)
    - Inverted NOx split (now 72% NO₂, 28% NO to suppress accumulation)
    - Reduced emissions (0.012/0.008 NOx, 0.028 VOC traffic)
    """
    no2, no, o3, voc = y
    solar = solar_arc(hour)
    bg = background_state(hour, traffic_density, solar_flux, wind_speed, temp_c, industrial, weekend_mode)
    mix = mixing_coeff(hour, solar_flux, wind_speed, inversion)
    temp_f = temperature_factor(temp_c)
    humid_f = humidity_factor(humidity)
    trap = trapping_factor(inversion, wind_speed)

    # Chemical reactions (rates in ppb/s)
    r_photolysis = j1(hour, solar_flux) * no2  # NO₂ + hν → NO + O
    r_titration = K_O3_NO * o3 * no            # O₃ + NO → NO₂ + O₂
    r_voc_chain = K_RO2_NO * no * math.sqrt(max(voc, 0.0)) * (solar ** 0.8) * temp_f * humid_f  # VOC radical
    r_voc_loss = K_VOC_LOSS * voc * (0.85 + 0.35 * temp_f)  # VOC oxidation
    r_humidity_o3_loss = K_O3_HUMIDITY_LOSS * o3 * humidity * (0.5 + solar)  # Humidity O₃ sink
    
    # NEW: NO₂ oxidation to HNO₃ (nighttime wet chemistry pathway)
    r_no2_hno3 = K_NO2_HNO3 * no2 * humidity * (1.0 - 0.7 * solar)  # More at night

    traffic_now = traffic_profile(hour, traffic_density, weekend_mode)
    # REDUCED emissions for realism
    e_nox = trap * (traffic_now * 0.010 + industrial * 0.006)
    e_voc = trap * (traffic_now * 0.028 + industrial * 0.040)
    release = K_TRAP_RELEASE * inversion * solar * (1.0 + temp_c / 50.0)

    # INVERTED NOx split: 72% NO₂ (was 28%), 28% NO (was 72%)
    # Reflects modern vehicle emissions: catalytic converters produce more NO₂ directly
    nox_to_no2_fraction = 0.72
    nox_to_no_fraction = 0.28

    return np.array(
        [
            # NO₂ time derivative
            np.clip(
                -r_photolysis              # Loss by photolysis
                + r_titration              # Gain from O₃ + NO
                + r_voc_chain              # Gain from VOC pathway
                + e_nox * nox_to_no2_fraction  # Emission (now majority species)
                + mix * (bg[0] - no2)      # Mixing toward background
                - K_NO2_DEP * no2          # ENHANCED: Dry deposition (3.6x stronger)
                - r_no2_hno3               # NEW: HNO₃ loss pathway
                - K_NO2_HNO3 * no2         # Additional HNO₃ removal
                - release * (no2 - bg[0]), # Inversion release
                -STATE_MAX[0],
                STATE_MAX[0],
            ),
            # NO time derivative
            np.clip(
                r_photolysis               # Production from NO₂ photolysis
                - r_titration              # Loss to O₃ reaction
                - r_voc_chain              # Loss in VOC pathway
                + e_nox * nox_to_no_fraction  # Emission (now minority species)
                + mix * (bg[1] - no)       # Mixing toward background
                - release * (no - bg[1]),  # Inversion release
                -STATE_MAX[1],
                STATE_MAX[1],
            ),
            # O₃ time derivative
            np.clip(
                r_photolysis               # O produced by NO₂ hν + O₂ → O₃
                - r_titration              # Loss to NO reaction
                + 0.72 * r_voc_chain       # Net O₃ from VOC pathway
                + mix * (bg[2] - o3)       # Mixing toward background
                - K_O3_DEP * o3            # Dry deposition
                - r_humidity_o3_loss       # Humidity-enhanced loss
                + 0.008 * industrial * solar * temp_f,  # REDUCED: Industrial O₃ source
                -STATE_MAX[2],
                STATE_MAX[2],
            ),
            # VOC time derivative
            np.clip(
                -r_voc_loss                # Loss via oxidation
                + e_voc                    # Emission
                + mix * (bg[3] - voc)      # Mixing toward background
                - 0.00005 * voc * humidity,  # Humidity-dependent removal
                -STATE_MAX[3],
                STATE_MAX[3],
            ),
        ],
        dtype=np.float64,
    )


def rk4_step(
    y: np.ndarray,
    dt: float,
    hour: float,
    traffic_density: float,
    solar_flux: float,
    wind_speed: float,
    temp_c: float,
    humidity: float,
    industrial: float,
    inversion: float,
    weekend_mode: float,
) -> np.ndarray:
    dh = dt / 3600.0
    k1 = derivatives(y, hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode)
    k2 = derivatives(clamp_state(y + 0.5 * dt * k1), (hour + 0.5 * dh) % 24.0, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode)
    k3 = derivatives(clamp_state(y + 0.5 * dt * k2), (hour + 0.5 * dh) % 24.0, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode)
    k4 = derivatives(clamp_state(y + dt * k3), (hour + dh) % 24.0, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode)
    return clamp_state(y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def advance_reference(
    y: np.ndarray,
    hour: float,
    traffic_density: float,
    solar_flux: float,
    wind_speed: float,
    temp_c: float,
    humidity: float,
    industrial: float,
    inversion: float,
    weekend_mode: float,
    dt: float = CHEM_DT,
) -> np.ndarray:
    substeps = max(1, math.ceil(dt / 1.0))
    sub_dt = dt / substeps
    for i in range(substeps):
        sub_hour = (hour + i * sub_dt / 3600.0) % 24.0
        y = rk4_step(y, sub_dt, sub_hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode)
    return clamp_state(y)


def sample_initial_state(
    hour: float,
    traffic_density: float,
    solar_flux: float,
    wind_speed: float,
    temp_c: float,
    humidity: float,
    industrial: float,
    inversion: float,
    weekend_mode: float,
) -> np.ndarray:
    baseline = urban_baseline(hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode)
    if np.random.rand() < 0.74:
        jitter = np.array(
            [
                np.random.uniform(0.50, 1.60),
                np.random.uniform(0.45, 1.75),
                np.random.uniform(0.35, 1.90),
                np.random.uniform(0.55, 1.75),
            ]
        )
        additive = np.array(
            [
                np.random.uniform(-5.0, 12.0),
                np.random.uniform(-6.0, 20.0),
                np.random.uniform(-4.0, 24.0),
                np.random.uniform(-14.0, 36.0),
            ]
        )
        return clamp_state(baseline * jitter + additive)

    return np.array(
        [
            np.random.uniform(0.0, 140.0),
            np.random.uniform(0.0, 140.0),
            np.random.uniform(0.0, 120.0),
            np.random.uniform(10.0, 520.0),
        ],
        dtype=np.float64,
    )


def build_dataset(cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build training dataset with optional multiple atmospheric scenarios."""
    x_rows: list[list[float]] = []
    y_rows: list[np.ndarray] = []
    
    # Divide samples across different atmospheric scenarios if enabled
    atmosphere_types = list(AtmosphereType) if cfg.use_multiple_atmospheres else [AtmosphereType.URBAN_HIGH_TRAFFIC]
    samples_per_type = cfg.samples // len(atmosphere_types)
    
    print(f"\n{'='*80}")
    print(f"Building training dataset with {cfg.samples} samples")
    if cfg.use_multiple_atmospheres:
        print(f"Using {len(atmosphere_types)} atmospheric scenarios")
        print(f"Samples per scenario: {samples_per_type}")
    print(f"{'='*80}\n")
    
    # Track statistics for status display
    stats = {atm.value: {"count": 0, "avg_traffic": 0.0, "avg_o3": 0.0, "avg_smog": 0.0} 
             for atm in atmosphere_types}

    for atm_idx, atmosphere_type in enumerate(atmosphere_types):
        for sample_idx in range(samples_per_type):
            if cfg.show_status and (sample_idx + 1) % max(1, samples_per_type // 5) == 0:
                progress = (atm_idx * samples_per_type + sample_idx + 1) / cfg.samples * 100
                print(f"[{progress:5.1f}%] {atmosphere_type.value}: {sample_idx + 1}/{samples_per_type} samples")
            
            # Sample parameters from the current atmospheric scenario
            hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode = \
                sample_atmospheric_params(atmosphere_type)
            
            hour_sin, hour_cos = hour_features(hour)

            y0 = sample_initial_state(hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode)
            y1 = advance_reference(y0, hour, traffic_density, solar_flux, wind_speed, temp_c, humidity, industrial, inversion, weekend_mode).astype(np.float32)

            x_rows.append(
                [
                    *y0.astype(np.float32),
                    traffic_density,
                    solar_flux,
                    wind_speed,
                    temp_c,
                    humidity,
                    industrial,
                    inversion,
                    weekend_mode,
                    hour_sin,
                    hour_cos,
                ]
            )
            y_rows.append(y1)
            
            # Update statistics
            smog_level = calculate_realistic_smog_index(y1[0], y1[2], y1[3], humidity)
            stats[atmosphere_type.value]["count"] += 1
            stats[atmosphere_type.value]["avg_traffic"] += traffic_density / samples_per_type
            stats[atmosphere_type.value]["avg_o3"] += y1[2] / samples_per_type
            stats[atmosphere_type.value]["avg_smog"] += smog_level / samples_per_type

    # Print final status summary
    print(f"\n{'='*80}")
    print("Dataset Statistics by Atmospheric Scenario:")
    print(f"{'='*80}")
    print(f"{'Scenario':<25} {'Samples':<12} {'Avg Traffic':<15} {'Avg O₃ (ppb)':<15} {'Avg Smog':<12}")
    print("-" * 79)
    for atm_type in atmosphere_types:
        stat = stats[atm_type.value]
        print(f"{atm_type.value:<25} {stat['count']:<12} "
              f"{stat['avg_traffic']:<15.3f} {stat['avg_o3']:<15.2f} {stat['avg_smog']:<12.3f}")
    print(f"{'='*80}\n")

    return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def split_dataset(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    permutation = np.random.permutation(len(x))
    x = x[permutation]
    y = y[permutation]
    cut = int(len(x) * 0.9)
    return x[:cut], x[cut:], y[:cut], y[cut:]


class SmogSurrogate(nn.Module):
    def __init__(self, x_mean: torch.Tensor, x_std: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor):
        super().__init__()
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)
        # Store state bounds for clamping
        self.register_buffer("state_min", torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32))
        self.register_buffer("state_max", torch.tensor([200.0, 150.0, 150.0, 700.0], dtype=torch.float32))
        
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, STATE_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = (x - self.x_mean) / (self.x_std + 1e-8)
        y_n = self.net(x_n)
        # Denormalize and clamp to realistic bounds
        y = y_n * (self.y_std + 1e-8) + self.y_mean
        # Strictly enforce state bounds to prevent unrealistic values
        y = torch.clamp(y, self.state_min, self.state_max)
        return y


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * xb.size(0)
            total_mae += torch.mean(torch.abs(pred - yb), dim=0).mean().item() * xb.size(0)
            count += xb.size(0)
    return total_loss / max(1, count), total_mae / max(1, count)


def train_model(cfg: TrainConfig) -> SmogSurrogate:
    x, y = build_dataset(cfg)
    x_train, x_val, y_train, y_val = split_dataset(x, y)
    print(f"Dataset: train={len(x_train)} val={len(x_val)}  dt={CHEM_DT:.0f}s")
    
    # Calculate and display correlations between traffic and smog
    train_traffic = x_train[:, 4]  # Traffic density column
    train_o3 = y_train[:, 2]  # O₃ output column
    train_humidity = x_train[:, 8]  # Humidity column
    
    # Calculate realistic smog indices for training set
    smog_indices = np.array([
        calculate_realistic_smog_index(y_train[i, 0], y_train[i, 2], y_train[i, 3], train_humidity[i])
        for i in range(len(y_train))
    ])
    
    # Correlation analysis
    traffic_smog_corr = np.corrcoef(train_traffic, smog_indices)[0, 1]
    traffic_o3_corr = np.corrcoef(train_traffic, train_o3)[0, 1]
    
    print(f"\nTraining Set Analysis:")
    print(f"  Traffic ↔ O₃ correlation: {traffic_o3_corr:.3f}")
    print(f"  Traffic ↔ Smog Index correlation: {traffic_smog_corr:.3f}")
    print(f"  Mean Smog Index: {np.mean(smog_indices):.3f}")
    print(f"  Mean O₃: {np.mean(train_o3):.1f} ppb")
    print(f"  Smog Index range: [{np.min(smog_indices):.3f}, {np.max(smog_indices):.3f}]")
    print()

    x_mean = torch.tensor(x_train.mean(axis=0), dtype=torch.float32)
    x_std = torch.tensor(x_train.std(axis=0) + 1e-8, dtype=torch.float32)
    y_mean = torch.tensor(y_train.mean(axis=0), dtype=torch.float32)
    y_std = torch.tensor(y_train.std(axis=0) + 1e-8, dtype=torch.float32)

    model = SmogSurrogate(x_mean, x_std, y_mean, y_std)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    
    print(f"{'='*80}")
    print(f"Training for {cfg.epochs} epochs (learning rate: {cfg.learning_rate})")
    print(f"{'='*80}\n")
    
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            
            # Base MSE loss
            loss = criterion(pred, yb)
            
            # Physics penalty: NOx production should be bounded
            nox_pred = pred[:, 0] + pred[:, 1]
            nox_in = xb[:, 0] + xb[:, 1]
            max_gain = 1.2 + xb[:, 4] * 1.8 + xb[:, 9] * 1.2  # Reduced max NOx gain
            nox_penalty = torch.relu(nox_pred - nox_in - max_gain).mean()
            
            # O3 should not exceed realistic bounds significantly
            o3_pred = pred[:, 2]
            o3_max_realistic = 150.0
            o3_penalty = torch.relu((o3_pred - o3_max_realistic) / o3_max_realistic).mean()
            
            # Regularization: encourage staying within reasonable ranges
            total_loss = loss + 0.02 * nox_penalty + 0.01 * o3_penalty
            
            optimiser.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        train_mse, train_mae = evaluate(model, train_loader, criterion)
        val_mse, val_mae = evaluate(model, val_loader, criterion)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs}  "
            f"train_mse={train_mse:.5f}  train_mae={train_mae:.4f}  "
            f"val_mse={val_mse:.5f}  val_mae={val_mae:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"Best validation MSE: {best_val:.5f}")
    print(f"{'='*80}\n")
    
    return model


def export_model(model: SmogSurrogate) -> None:
    model.eval()
    dummy = torch.zeros(1, INPUT_DIM)
    torch.onnx.export(
        model,
        dummy,
        "smog_surrogate.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print("Saved: smog_surrogate.onnx")


def main() -> None:
    model = train_model(CFG)
    export_model(model)
    
    # Demonstrate the realistic smog index calculation
    print("\nRealistic Smog Index Function Demonstration:")
    print("=" * 70)
    print("Testing different pollution scenarios:\n")
    
    scenarios = [
        ("Clean air (low NO₂, low O₃)", 5.0, 15.0, 50.0, 0.40),
        ("Light pollution", 30.0, 40.0, 200.0, 0.50),
        ("Moderate smog", 80.0, 120.0, 400.0, 0.60),
        ("Heavy smog", 200.0, 180.0, 700.0, 0.75),
        ("High pollution (hot, humid)", 150.0, 200.0, 600.0, 0.90),
    ]
    
    print(f"{'Scenario':<35} {'NO₂':<8} {'O₃':<8} {'VOC':<8} {'Humidity':<10} {'Smog Index':<12}")
    print("-" * 81)
    
    for name, no2, o3, voc, humidity in scenarios:
        smog = calculate_realistic_smog_index(no2, o3, voc, humidity)
        print(f"{name:<35} {no2:<8.1f} {o3:<8.1f} {voc:<8.1f} {humidity:<10.2f} {smog:<12.3f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
