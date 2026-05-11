import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CHEM_DT = 10.0
STATE_DIM = 4
INPUT_DIM = 14


@dataclass(frozen=True)
class TrainConfig:
    samples: int = int(os.getenv("SMOG_TRAIN_SAMPLES", "45000"))
    epochs: int = int(os.getenv("SMOG_EPOCHS", "32"))
    batch_size: int = int(os.getenv("SMOG_BATCH_SIZE", "512"))
    learning_rate: float = float(os.getenv("SMOG_LR", "0.001"))
    seed: int = int(os.getenv("SMOG_SEED", "42"))


CFG = TrainConfig()
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

try:
    import onnx  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'onnx'. Install it in the same venv with `pip install onnx` before training."
    ) from exc

K_O3_NO = 2.5e-2
K_RO2_NO = 2.2e-3
K_VOC_LOSS = 6.0e-5
J1_MAX = 7.5e-3
K_O3_DEP = 2.0e-4
K_NO2_DEP = 7.0e-5
K_MIX_BASE = 6.0e-5
K_MIX_DAY = 2.2e-4
K_MIX_WIND = 2.4e-4
K_O3_HUMIDITY_LOSS = 8.5e-5
K_TRAP_RELEASE = 1.2e-4

STATE_MAX = np.array([400.0, 330.0, 260.0, 1100.0], dtype=np.float64)


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
    traffic_now = effective_traffic_density(hour, traffic_density, weekend_mode)
    weekend_bias = 0.85 if weekend_mode >= 0.5 else 1.0
    return np.array(
        [
            np.clip(2.0 + 2.4 * gaussian_wrap(hour, 7.0, 1.8) + 5.0 * industrial * weekend_bias, 0.5, 35.0),
            np.clip(0.2 + 0.85 * gaussian_wrap(hour, 8.0, 1.0) + 0.65 * gaussian_wrap(hour, 18.0, 1.2), 0.0, 10.0),
            np.clip(10.0 + 16.0 * solar_arc(hour) * solar_flux + 11.0 * wind_speed + 3.0 * temp_c / 30.0, 4.0, 70.0),
            np.clip(14.0 + 10.0 * traffic_now + 28.0 * industrial, 8.0, 130.0),
        ],
        dtype=np.float64,
    )


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
    rush = traffic_profile(hour, traffic_density, weekend_mode)
    day = solar_arc(hour) * solar_flux
    trap = trapping_factor(inversion, wind_speed)
    return np.array(
        [
            np.clip(12.0 + 19.0 * rush + 10.0 * industrial * trap, 3.0, 150.0),
            np.clip(8.0 + 25.0 * rush * trap, 1.0, 140.0),
            np.clip(6.0 + 16.0 * day + 6.0 * wind_speed + 5.0 * industrial, 0.0, 90.0),
            np.clip(
                38.0 + 80.0 * rush + 85.0 * industrial + 4.0 * humidity + 0.8 * max(0.0, temp_c - 20.0),
                15.0,
                420.0,
            ),
        ],
        dtype=np.float64,
    )


def clamp_state(y: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(y, 0.0), STATE_MAX)


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
    no2, no, o3, voc = y
    solar = solar_arc(hour)
    bg = background_state(hour, traffic_density, solar_flux, wind_speed, temp_c, industrial, weekend_mode)
    mix = mixing_coeff(hour, solar_flux, wind_speed, inversion)
    temp_f = temperature_factor(temp_c)
    humid_f = humidity_factor(humidity)
    trap = trapping_factor(inversion, wind_speed)

    r_photolysis = j1(hour, solar_flux) * no2
    r_titration = K_O3_NO * o3 * no
    r_voc_chain = K_RO2_NO * no * math.sqrt(max(voc, 0.0)) * (solar ** 0.8) * temp_f * humid_f
    r_voc_loss = K_VOC_LOSS * voc * (0.85 + 0.35 * temp_f)
    r_humidity_o3_loss = K_O3_HUMIDITY_LOSS * o3 * humidity * (0.5 + solar)

    traffic_now = traffic_profile(hour, traffic_density, weekend_mode)
    e_nox = trap * (traffic_now * 0.030 + industrial * 0.018)
    e_voc = trap * (traffic_now * 0.095 + industrial * 0.110)
    release = K_TRAP_RELEASE * inversion * solar * (1.0 + temp_c / 50.0)

    return np.array(
        [
            np.clip(
                -r_photolysis + r_titration + r_voc_chain + e_nox * 0.28 + mix * (bg[0] - no2) - K_NO2_DEP * no2 - release * (no2 - bg[0]),
                -STATE_MAX[0],
                STATE_MAX[0],
            ),
            np.clip(
                r_photolysis - r_titration - r_voc_chain + e_nox * 0.72 + mix * (bg[1] - no) - release * (no - bg[1]),
                -STATE_MAX[1],
                STATE_MAX[1],
            ),
            np.clip(
                r_photolysis - r_titration + 0.72 * r_voc_chain + mix * (bg[2] - o3) - K_O3_DEP * o3 - r_humidity_o3_loss + 0.010 * industrial * solar * temp_f,
                -STATE_MAX[2],
                STATE_MAX[2],
            ),
            np.clip(
                -r_voc_loss + e_voc + mix * (bg[3] - voc) - 0.00005 * voc * humidity,
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
    x_rows: list[list[float]] = []
    y_rows: list[np.ndarray] = []

    for _ in range(cfg.samples):
        traffic_density = np.random.uniform(0.0, 1.0)
        solar_flux = np.random.uniform(0.2, 1.5)
        wind_speed = np.random.uniform(0.0, 1.0)
        temp_c = np.random.uniform(10.0, 40.0)
        humidity = np.random.uniform(0.1, 1.0)
        industrial = np.random.uniform(0.0, 1.0)
        inversion = np.random.uniform(0.0, 1.0)
        weekend_mode = 1.0 if np.random.rand() < 0.18 else 0.0
        hour = np.random.uniform(0.0, 24.0)
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
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, STATE_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = (x - self.x_mean) / self.x_std
        y_n = self.net(x_n)
        y = y_n * self.y_std + self.y_mean
        return torch.relu(y)


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
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            nox_pred = pred[:, 0] + pred[:, 1]
            nox_in = xb[:, 0] + xb[:, 1]
            max_gain = 2.0 + xb[:, 4] * 2.5 + xb[:, 9] * 2.0
            nox_penalty = torch.relu(nox_pred - nox_in - max_gain).mean()
            loss = criterion(pred, yb) + 0.03 * nox_penalty
            optimiser.zero_grad()
            loss.backward()
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


if __name__ == "__main__":
    main()
