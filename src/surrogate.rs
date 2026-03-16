use ort::session::Session;
use ort::value::Tensor;

use crate::chemistry::{ChemState, SmogParams};

pub struct NNSurrogate {
    session: Session,
}

impl NNSurrogate {
    pub fn load(path: &str) -> Option<Self> {
        let session = Session::builder().ok()?.commit_from_file(path).ok()?;
        Some(Self { session })
    }

    /// Predict one fixed 10-second chemistry step.
    pub fn predict(&mut self, state: &ChemState, params: &SmogParams, hour: f64) -> Option<ChemState> {
        let data: Vec<f32> = vec![
            state.no2 as f32,
            state.no as f32,
            state.o3 as f32,
            state.voc as f32,
            params.traffic_density as f32,
            params.solar_flux as f32,
            params.wind_speed as f32,
            params.temperature_c as f32,
            params.humidity as f32,
            params.industrial_emissions as f32,
            params.inversion_strength as f32,
            if params.weekend_mode { 1.0 } else { 0.0 },
            hour as f32,
        ];

        let ort_input = Tensor::<f32>::from_array(([1usize, 13usize], data)).ok()?;
        let outputs = self.session.run(ort::inputs!["input" => ort_input]).ok()?;
        let (_, raw) = outputs["output"].try_extract_tensor::<f32>().ok()?;

        Some(ChemState {
            no2: (raw[0] as f64).max(0.0),
            no: (raw[1] as f64).max(0.0),
            o3: (raw[2] as f64).max(0.0),
            voc: (raw[3] as f64).max(0.0),
        })
    }
}

#[derive(bevy::prelude::Resource, Default)]
pub struct SurrogateRes {
    pub inner: Option<NNSurrogate>,
    pub enabled: bool,
}
