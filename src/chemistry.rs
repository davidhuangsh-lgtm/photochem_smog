/// Expanded photochemical smog box model used by the live exhibit.
///
/// Units:
/// - concentrations in ppb
/// - temperature in °C
/// - humidity / sliders are dimensionless fractions
/// - time in seconds (integration) or decimal hours (solar / traffic drivers)
///
/// Educational additions beyond the original version:
/// - industrial VOC / NOx source strength,
/// - humidity-dependent radical efficiency,
/// - temperature-dependent chemistry acceleration,
/// - boundary-layer / inversion trapping,
/// - weekend mode that reduces traffic while preserving photochemistry.

#[derive(Clone, Debug)]
pub struct ChemState {
    pub no2: f64,
    pub no: f64,
    pub o3: f64,
    pub voc: f64,
}

#[derive(Clone, Debug)]
pub struct SmogParams {
    pub traffic_density: f64,      // 0.0 .. 1.0
    pub solar_flux: f64,           // 0.2 .. 1.5
    pub wind_speed: f64,           // 0.0 .. 1.0
    pub temperature_c: f64,        // 10 .. 40
    pub humidity: f64,             // 0.1 .. 1.0
    pub industrial_emissions: f64, // 0.0 .. 1.0
    pub inversion_strength: f64,   // 0.0 .. 1.0
    pub weekend_mode: bool,
}

impl Default for SmogParams {
    fn default() -> Self {
        Self {
            traffic_density: 0.55,
            solar_flux: 1.0,
            wind_speed: 0.30,
            temperature_c: 28.0,
            humidity: 0.45,
            industrial_emissions: 0.25,
            inversion_strength: 0.20,
            weekend_mode: false,
        }
    }
}

pub const CHEM_DT: f64 = 10.0;

impl ChemState {
    /// Stylized but plausible urban baseline for the current controls.
    pub fn urban_baseline(hour: f64, p: &SmogParams) -> Self {
        let rush = traffic_profile(hour, p.traffic_density, p.weekend_mode);
        let day = solar_arc(hour) * p.solar_flux;
        let trap = trapping_factor(p.inversion_strength, p.wind_speed);
        let industry = p.industrial_emissions;
        Self {
            no2: (12.0 + 19.0 * rush + 10.0 * industry * trap).clamp(3.0, 150.0),
            no: (8.0 + 25.0 * rush * trap).clamp(1.0, 140.0),
            o3: (6.0 + 16.0 * day + 6.0 * p.wind_speed + 5.0 * industry).clamp(0.0, 90.0),
            voc: (38.0 + 80.0 * rush + 85.0 * industry).clamp(15.0, 420.0),
        }
    }
}

const K_O3_NO: f64 = 2.5e-2;
const K_RO2_NO: f64 = 2.2e-3;
const K_VOC_LOSS: f64 = 6.0e-5;
const J1_MAX: f64 = 7.5e-3;
const K_O3_DEP: f64 = 2.0e-4;
const K_NO2_DEP: f64 = 7.0e-5;
const K_MIX_BASE: f64 = 6.0e-5;
const K_MIX_DAY: f64 = 2.2e-4;
const K_MIX_WIND: f64 = 2.4e-4;
const K_O3_HUMIDITY_LOSS: f64 = 8.5e-5;
const K_TRAP_RELEASE: f64 = 1.2e-4;

const MAX_NO2: f64 = 400.0;
const MAX_NO: f64 = 330.0;
const MAX_O3: f64 = 260.0;
const MAX_VOC: f64 = 1100.0;

fn gaussian_wrap(hour: f64, center: f64, width: f64) -> f64 {
    let raw = (hour - center).abs();
    let d = raw.min(24.0 - raw);
    (-0.5 * (d / width).powi(2)).exp()
}

pub fn solar_arc(hour: f64) -> f64 {
    if !(6.0..=18.0).contains(&hour) {
        return 0.0;
    }
    ((std::f64::consts::PI * (hour - 6.0) / 12.0).sin()).max(0.0)
}

pub fn j1(hour: f64, solar_flux: f64) -> f64 {
    J1_MAX * solar_arc(hour).powf(1.25) * solar_flux
}

pub fn traffic_profile(hour: f64, traffic_density: f64, weekend_mode: bool) -> f64 {
    let weekend_scale = if weekend_mode { 0.68 } else { 1.0 };
    let base = (0.22 + 0.42 * traffic_density) * if weekend_mode { 0.78 } else { 1.0 };
    let morning = (0.58 + 0.92 * traffic_density)
        * gaussian_wrap(hour, 8.0, if weekend_mode { 1.3 } else { 0.9 });
    let evening = (0.44 + 0.74 * traffic_density)
        * gaussian_wrap(hour, 17.6, if weekend_mode { 1.4 } else { 1.1 });
    let midday = (0.10 + 0.18 * traffic_density) * gaussian_wrap(hour, 13.0, 2.4);
    weekend_scale * (base + morning + evening + midday)
}

pub fn mixing_coeff(hour: f64, solar_flux: f64, wind_speed: f64, inversion_strength: f64) -> f64 {
    let midday_phase = ((hour - 8.5) / 8.5).clamp(0.0, 1.0);
    let mixed_day = (std::f64::consts::PI * midday_phase).sin().max(0.0);
    let inversion_drag = 1.0 - 0.72 * inversion_strength;
    (K_MIX_BASE + K_MIX_DAY * mixed_day * solar_flux + K_MIX_WIND * wind_speed)
        * inversion_drag.max(0.18)
}

pub fn trapping_factor(inversion_strength: f64, wind_speed: f64) -> f64 {
    (1.0 + 1.2 * inversion_strength - 0.55 * wind_speed).clamp(0.55, 2.1)
}

pub fn temperature_factor(temp_c: f64) -> f64 {
    (0.82 + 0.018 * (temp_c - 18.0)).clamp(0.65, 1.45)
}

pub fn humidity_factor(humidity: f64) -> f64 {
    (0.82 + 0.36 * humidity).clamp(0.75, 1.22)
}

pub fn background_state(hour: f64, p: &SmogParams) -> ChemState {
    let weekend_bias = if p.weekend_mode { 0.85 } else { 1.0 };
    ChemState {
        no2: (2.0
            + 2.2 * gaussian_wrap(hour, 7.0, 1.8)
            + 5.0 * p.industrial_emissions * weekend_bias)
            .clamp(0.5, 35.0),
        no: (0.3 + 1.0 * gaussian_wrap(hour, 8.0, 1.0) + 0.8 * gaussian_wrap(hour, 18.0, 1.2))
            .clamp(0.0, 10.0),
        o3: (10.0
            + 16.0 * solar_arc(hour) * p.solar_flux
            + 11.0 * p.wind_speed
            + 3.0 * p.temperature_c / 30.0)
            .clamp(4.0, 70.0),
        voc: (16.0 + 9.0 * p.traffic_density + 28.0 * p.industrial_emissions).clamp(8.0, 130.0),
    }
}

pub fn derivatives(y: &ChemState, hour: f64, p: &SmogParams) -> ChemState {
    let solar = solar_arc(hour);
    let bg = background_state(hour, p);
    let mix = mixing_coeff(hour, p.solar_flux, p.wind_speed, p.inversion_strength);
    let temp_f = temperature_factor(p.temperature_c);
    let humid_f = humidity_factor(p.humidity);
    let trap = trapping_factor(p.inversion_strength, p.wind_speed);

    let r_photolysis = j1(hour, p.solar_flux) * y.no2;
    let r_titration = K_O3_NO * y.o3 * y.no;
    let r_voc_chain = K_RO2_NO * y.no * y.voc.max(0.0).sqrt() * solar.powf(0.8) * temp_f * humid_f;
    let r_voc_loss = K_VOC_LOSS * y.voc * (0.85 + 0.35 * temp_f);
    let r_humidity_o3_loss = K_O3_HUMIDITY_LOSS * y.o3 * p.humidity * (0.5 + solar);

    let traffic_now = traffic_profile(hour, p.traffic_density, p.weekend_mode);
    let industrial = p.industrial_emissions;
    let e_nox = trap * (traffic_now * 0.030 + industrial * 0.018);
    let e_voc = trap * (traffic_now * 0.095 + industrial * 0.110);
    let release = K_TRAP_RELEASE * p.inversion_strength * solar * (1.0 + p.temperature_c / 50.0);

    ChemState {
        no2: (-r_photolysis + r_titration + r_voc_chain + e_nox * 0.28 + mix * (bg.no2 - y.no2)
            - K_NO2_DEP * y.no2
            - release * (y.no2 - bg.no2))
            .clamp(-MAX_NO2, MAX_NO2),
        no: (r_photolysis - r_titration - r_voc_chain + e_nox * 0.72 + mix * (bg.no - y.no)
            - release * (y.no - bg.no))
            .clamp(-MAX_NO, MAX_NO),
        o3: (r_photolysis - r_titration + 0.72 * r_voc_chain + mix * (bg.o3 - y.o3)
            - K_O3_DEP * y.o3
            - r_humidity_o3_loss
            + 0.010 * industrial * solar * temp_f)
            .clamp(-MAX_O3, MAX_O3),
        voc: (-r_voc_loss + e_voc + mix * (bg.voc - y.voc) - 0.00005 * y.voc * p.humidity)
            .clamp(-MAX_VOC, MAX_VOC),
    }
}

pub fn clamp_state(y: &ChemState) -> ChemState {
    ChemState {
        no2: y.no2.clamp(0.0, MAX_NO2),
        no: y.no.clamp(0.0, MAX_NO),
        o3: y.o3.clamp(0.0, MAX_O3),
        voc: y.voc.clamp(0.0, MAX_VOC),
    }
}

pub fn step_rk4(y: &ChemState, dt: f64, hour: f64, p: &SmogParams) -> ChemState {
    let dh = dt / 3600.0;
    let k1 = derivatives(y, hour, p);

    let y2 = clamp_state(&ChemState {
        no2: y.no2 + 0.5 * dt * k1.no2,
        no: y.no + 0.5 * dt * k1.no,
        o3: y.o3 + 0.5 * dt * k1.o3,
        voc: y.voc + 0.5 * dt * k1.voc,
    });
    let k2 = derivatives(&y2, (hour + 0.5 * dh) % 24.0, p);

    let y3 = clamp_state(&ChemState {
        no2: y.no2 + 0.5 * dt * k2.no2,
        no: y.no + 0.5 * dt * k2.no,
        o3: y.o3 + 0.5 * dt * k2.o3,
        voc: y.voc + 0.5 * dt * k2.voc,
    });
    let k3 = derivatives(&y3, (hour + 0.5 * dh) % 24.0, p);

    let y4 = clamp_state(&ChemState {
        no2: y.no2 + dt * k3.no2,
        no: y.no + dt * k3.no,
        o3: y.o3 + dt * k3.o3,
        voc: y.voc + dt * k3.voc,
    });
    let k4 = derivatives(&y4, (hour + dh) % 24.0, p);

    clamp_state(&ChemState {
        no2: y.no2 + (dt / 6.0) * (k1.no2 + 2.0 * k2.no2 + 2.0 * k3.no2 + k4.no2),
        no: y.no + (dt / 6.0) * (k1.no + 2.0 * k2.no + 2.0 * k3.no + k4.no),
        o3: y.o3 + (dt / 6.0) * (k1.o3 + 2.0 * k2.o3 + 2.0 * k3.o3 + k4.o3),
        voc: y.voc + (dt / 6.0) * (k1.voc + 2.0 * k2.voc + 2.0 * k3.voc + k4.voc),
    })
}
