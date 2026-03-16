use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin};

mod chemistry;
mod surrogate;

use chemistry::{
    humidity_factor, j1, mixing_coeff, solar_arc, step_rk4, temperature_factor, traffic_profile,
    trapping_factor, ChemState, SmogParams, CHEM_DT,
};
use surrogate::SurrogateRes;

const HISTORY_LEN: usize = 420;
const MAX_CHEM_STEPS_PER_FRAME: usize = 24;
const CAR_POOL: usize = 28;
const TRAFFIC_SPIKE_DURATION_HOURS: f32 = 1.0;
const TRAFFIC_SPIKE_BOOST: f64 = 0.24;
const RESET_WARMUP_SECONDS: f64 = 120.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Photochemical Smog — AI-Lab × ChemiClub".into(),
                resolution: (1440.0, 820.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .init_resource::<SimState>()
        .insert_resource(SurrogateRes::default())
        .add_systems(Startup, (load_surrogate, setup_scene))
        .add_systems(Update, (advance_chemistry, sync_visuals, render_ui).chain())
        .run();
}

#[derive(Clone, Copy)]
struct HistoryPoint {
    o3: f32,
    no2: f32,
    no: f32,
    voc: f32,
}

#[derive(Resource)]
struct SimState {
    chem: ChemState,
    params: SmogParams,
    time_of_day: f32,
    auto_advance: bool,
    paused: bool,
    traffic_spike_remaining_hours: f32,
    clock_speed: f32,
    step_accumulator: f64,
    history: VecDeque<HistoryPoint>,
}

impl Default for SimState {
    fn default() -> Self {
        let params = SmogParams::default();
        let time_of_day = 7.2;
        let chem = seeded_atmosphere(time_of_day as f64, &params);
        let mut history = VecDeque::with_capacity(HISTORY_LEN);
        history.push_back(HistoryPoint {
            o3: chem.o3 as f32,
            no2: chem.no2 as f32,
            no: chem.no as f32,
            voc: chem.voc as f32,
        });
        Self {
            chem,
            params,
            time_of_day,
            auto_advance: true,
            paused: false,
            traffic_spike_remaining_hours: 0.0,
            clock_speed: 180.0,
            step_accumulator: 0.0,
            history,
        }
    }
}

impl SimState {
    fn traffic_spike_active(&self) -> bool {
        self.traffic_spike_remaining_hours > 0.0
    }

    fn active_traffic_density(&self) -> f64 {
        (self.params.traffic_density
            + if self.traffic_spike_active() {
                TRAFFIC_SPIKE_BOOST
            } else {
                0.0
            })
        .clamp(0.0, 1.0)
    }

    fn effective_params(&self) -> SmogParams {
        let mut params = self.params.clone();
        params.traffic_density = self.active_traffic_density();
        params
    }

    fn start_traffic_spike(&mut self) {
        self.traffic_spike_remaining_hours = TRAFFIC_SPIKE_DURATION_HOURS;
    }

    fn tick_interventions(&mut self) {
        if self.traffic_spike_remaining_hours > 0.0 {
            self.traffic_spike_remaining_hours =
                (self.traffic_spike_remaining_hours - (CHEM_DT as f32 / 3600.0)).max(0.0);
        }
    }

    fn clear_interventions(&mut self) {
        self.traffic_spike_remaining_hours = 0.0;
    }

    fn reset_atmosphere(&mut self) {
        let params = self.effective_params();
        self.chem = seeded_atmosphere(self.time_of_day as f64, &params);
        self.step_accumulator = 0.0;
        self.history.clear();
        self.push_history_point();
    }

    fn push_history_point(&mut self) {
        self.history.push_back(HistoryPoint {
            o3: self.chem.o3 as f32,
            no2: self.chem.no2 as f32,
            no: self.chem.no as f32,
            voc: self.chem.voc as f32,
        });
        while self.history.len() > HISTORY_LEN {
            self.history.pop_front();
        }
    }
}

fn seeded_atmosphere(time_of_day: f64, params: &SmogParams) -> ChemState {
    let warmup_steps = (RESET_WARMUP_SECONDS / CHEM_DT).round() as usize;
    let warmup_hours = RESET_WARMUP_SECONDS / 3600.0;
    let mut hour = (time_of_day - warmup_hours).rem_euclid(24.0);
    let mut state = ChemState::urban_baseline(hour, params);

    for _ in 0..warmup_steps {
        state = step_rk4(&state, CHEM_DT, hour, params);
        hour = (hour + CHEM_DT / 3600.0).rem_euclid(24.0);
    }

    state
}

#[derive(Component)]
struct SmogLayer;
#[derive(Component)]
struct No2Layer;
#[derive(Component)]
struct SkyBg;
#[derive(Component)]
struct SunMarker;
#[derive(Component)]
struct CloudLayer;
#[derive(Component)]
struct BuildingMarker {
    shade: f32,
}
#[derive(Component)]
struct BuildingWindow {
    glow_seed: f32,
}
#[derive(Component)]
struct CarMarker {
    dir: f32,
    slot: usize,
}
#[derive(Component, Clone, Copy)]
struct CarLight {
    kind: CarLightKind,
}

#[derive(Clone, Copy)]
enum CarLightKind {
    Headlight,
    Taillight,
}

fn load_surrogate(mut res: ResMut<SurrogateRes>) {
    use surrogate::NNSurrogate;
    match NNSurrogate::load("smog_surrogate.onnx") {
        Some(nn) => {
            res.inner = Some(nn);
            res.enabled = true;
            println!("[surrogate] smog_surrogate.onnx loaded — neural solver active");
        }
        None => {
            println!("[surrogate] smog_surrogate.onnx not found — falling back to RK4");
        }
    }
}

fn setup_scene(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((
        Sprite {
            color: Color::srgb(0.44, 0.67, 0.90),
            custom_size: Some(Vec2::new(1600.0, 900.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
        SkyBg,
    ));

    commands.spawn((
        Sprite {
            color: Color::srgba(0.96, 0.97, 0.99, 0.0),
            custom_size: Some(Vec2::new(1500.0, 260.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 220.0, 0.2),
        CloudLayer,
    ));

    commands.spawn((
        Sprite {
            color: Color::srgba(0.88, 0.80, 0.22, 0.0),
            custom_size: Some(Vec2::new(1400.0, 260.0)),
            ..default()
        },
        Transform::from_xyz(0.0, -70.0, 1.0),
        SmogLayer,
    ));

    commands.spawn((
        Sprite {
            color: Color::srgba(0.78, 0.32, 0.10, 0.0),
            custom_size: Some(Vec2::new(1400.0, 160.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 85.0, 1.1),
        No2Layer,
    ));

    commands.spawn((
        Sprite {
            color: Color::srgb(0.18, 0.18, 0.20),
            custom_size: Some(Vec2::new(1500.0, 100.0)),
            ..default()
        },
        Transform::from_xyz(0.0, -330.0, 2.0),
    ));

    let buildings: &[(f32, f32, f32)] = &[
        (-560.0, 190.0, 80.0),
        (-430.0, 270.0, 86.0),
        (-275.0, 330.0, 92.0),
        (-120.0, 210.0, 74.0),
        (40.0, 370.0, 102.0),
        (205.0, 240.0, 86.0),
        (365.0, 300.0, 90.0),
        (525.0, 215.0, 76.0),
    ];
    for (index, &(x, h, w)) in buildings.iter().enumerate() {
        spawn_building(&mut commands, index, x, h, w);
    }

    commands.spawn((
        Sprite {
            color: Color::srgb(1.0, 0.95, 0.60),
            custom_size: Some(Vec2::new(64.0, 64.0)),
            ..default()
        },
        Transform::from_xyz(-380.0, 180.0, 0.5),
        SunMarker,
    ));

    for slot in 0..CAR_POOL {
        let lane_index = slot / 2;
        let x = -760.0 + lane_index as f32 * 118.0;
        let dir = if slot % 2 == 0 { 1.0_f32 } else { -1.0 };
        let y = if dir > 0.0 { -298.0 } else { -315.0 };
        spawn_car(&mut commands, slot, x, y, dir);
    }
}

fn spawn_building(commands: &mut Commands, index: usize, x: f32, h: f32, w: f32) {
    let cols = ((w / 20.0).floor() as usize).max(2);
    let rows = ((h / 34.0).floor() as usize).max(3);
    let x_step = (w - 26.0).max(14.0) / cols as f32;
    let y_step = (h - 28.0).max(18.0) / rows as f32;
    let window_w = (x_step * 0.42).clamp(7.0, 14.0);
    let window_h = (y_step * 0.44).clamp(10.0, 16.0);
    let shade = 0.92 + index as f32 * 0.04;

    commands
        .spawn((
            Sprite {
                color: Color::srgb(0.17, 0.18, 0.22),
                custom_size: Some(Vec2::new(w, h)),
                ..default()
            },
            Transform::from_xyz(x, -285.0 + h * 0.5, 3.0),
            BuildingMarker { shade },
        ))
        .with_children(|parent| {
            let x_start = -w * 0.5 + 14.0;
            let y_start = -h * 0.5 + 18.0;
            for row in 0..rows {
                for col in 0..cols {
                    parent.spawn((
                        Sprite {
                            color: Color::srgba(1.0, 0.86, 0.58, 0.0),
                            custom_size: Some(Vec2::new(window_w, window_h)),
                            ..default()
                        },
                        Transform::from_xyz(
                            x_start + col as f32 * x_step,
                            y_start + row as f32 * y_step,
                            0.2,
                        ),
                        BuildingWindow {
                            glow_seed: 1.7 * index as f32 + 2.3 * row as f32 + 1.1 * col as f32,
                        },
                    ));
                }
            }
        });
}

fn spawn_car(commands: &mut Commands, slot: usize, x: f32, y: f32, dir: f32) {
    commands
        .spawn((
            Sprite {
                color: Color::srgba(0.62, 0.13, 0.12, 0.80),
                custom_size: Some(Vec2::new(46.0, 16.0)),
                ..default()
            },
            Transform::from_xyz(x, y, 4.0),
            CarMarker { dir, slot },
        ))
        .with_children(|parent| {
            let head_x = if dir > 0.0 { 20.0 } else { -20.0 };
            let tail_x = -head_x;
            for offset in [-3.5_f32, 3.5_f32] {
                parent.spawn((
                    Sprite {
                        color: Color::srgba(1.0, 0.95, 0.82, 0.0),
                        custom_size: Some(Vec2::new(5.0, 3.0)),
                        ..default()
                    },
                    Transform::from_xyz(head_x, offset, 0.2),
                    CarLight {
                        kind: CarLightKind::Headlight,
                    },
                ));
                parent.spawn((
                    Sprite {
                        color: Color::srgba(0.98, 0.18, 0.16, 0.0),
                        custom_size: Some(Vec2::new(4.5, 3.0)),
                        ..default()
                    },
                    Transform::from_xyz(tail_x, offset, 0.2),
                    CarLight {
                        kind: CarLightKind::Taillight,
                    },
                ));
            }
        });
}

fn advance_chemistry(
    mut state: ResMut<SimState>,
    mut surrogate: ResMut<SurrogateRes>,
    time: Res<Time>,
) {
    if state.paused {
        return;
    }

    state.step_accumulator += time.delta_secs_f64() * state.clock_speed as f64;
    let max_accumulator = CHEM_DT * MAX_CHEM_STEPS_PER_FRAME as f64;
    if state.step_accumulator > max_accumulator {
        state.step_accumulator = max_accumulator;
    }

    let mut steps = 0usize;
    while state.step_accumulator >= CHEM_DT && steps < MAX_CHEM_STEPS_PER_FRAME {
        let params = state.effective_params();
        let hour = state.time_of_day as f64;
        let prev = state.chem.clone();

        let next = if surrogate.enabled {
            surrogate
                .inner
                .as_mut()
                .and_then(|nn| nn.predict(&prev, &params, hour))
                .unwrap_or_else(|| step_rk4(&prev, CHEM_DT, hour, &params))
        } else {
            step_rk4(&prev, CHEM_DT, hour, &params)
        };

        state.chem = next;
        if state.auto_advance {
            state.time_of_day += (CHEM_DT / 3600.0) as f32;
            if state.time_of_day >= 24.0 {
                state.time_of_day -= 24.0;
            }
        }
        state.tick_interventions();
        state.step_accumulator -= CHEM_DT;
        state.push_history_point();
        steps += 1;
    }
}

fn sync_visuals(
    state: Res<SimState>,
    time: Res<Time>,
    mut visuals: ParamSet<(
        Query<&mut Sprite, With<SkyBg>>,
        Query<&mut Sprite, With<CloudLayer>>,
        Query<(&mut Sprite, &mut Transform), With<SmogLayer>>,
        Query<(&mut Sprite, &mut Transform), With<No2Layer>>,
        Query<(&mut Transform, &mut Sprite), With<SunMarker>>,
        Query<(&CarMarker, &mut Transform, &mut Sprite, &mut Visibility)>,
        Query<
            (
                &mut Sprite,
                Option<&BuildingMarker>,
                Option<&BuildingWindow>,
                Option<&CarLight>,
            ),
            Or<(With<BuildingMarker>, With<BuildingWindow>, With<CarLight>)>,
        >,
    )>,
) {
    let o3_ppb = (state.chem.o3 as f32).clamp(0.0, 260.0);
    let no2_ppb = (state.chem.no2 as f32).clamp(0.0, 220.0);
    let hour = state.time_of_day as f64;
    let day_f = solar_arc(hour) as f32 * state.params.solar_flux as f32;
    let wind_f = state.params.wind_speed as f32;
    let active_density = state.active_traffic_density();
    let traffic_now = traffic_profile(hour, active_density, state.params.weekend_mode) as f32;
    let trap = trapping_factor(state.params.inversion_strength, state.params.wind_speed) as f32;
    let humidity = state.params.humidity as f32;
    let smog_f = (o3_ppb / 155.0).clamp(0.0, 1.0);
    let no2_f = (no2_ppb / 120.0).clamp(0.0, 1.0);
    let cloudiness =
        (((1.12 - state.params.solar_flux as f32) / 0.92) + 0.25 * humidity).clamp(0.0, 1.0);
    let ambient_brightness = ((0.08 + 0.92 * day_f.clamp(0.0, 1.0))
        * (1.0 - 0.42 * smog_f - 0.20 * no2_f - 0.18 * cloudiness))
        .clamp(0.06, 1.0);
    let darkness = 1.0 - ambient_brightness;

    {
        let mut sky_q = visuals.p0();
        if let Ok(mut s) = sky_q.get_single_mut() {
            let humid_tint = 0.08 * humidity;
            s.color = Color::srgb(
                0.08 + 0.26 * ambient_brightness + 0.16 * smog_f,
                0.12 + 0.49 * ambient_brightness - 0.12 * smog_f + humid_tint,
                0.18 + 0.54 * ambient_brightness - 0.20 * smog_f + humid_tint,
            );
        }
    }

    {
        let mut cloud_q = visuals.p1();
        if let Ok(mut c) = cloud_q.get_single_mut() {
            c.color = Color::srgba(0.96, 0.97, 0.99, 0.06 + 0.36 * cloudiness);
        }
    }

    {
        let mut smog_q = visuals.p2();
        if let Ok((mut s, mut t)) = smog_q.get_single_mut() {
            let alpha =
                ((o3_ppb / 135.0) * (1.0 - 0.28 * wind_f) * (0.75 + 0.25 * trap)).clamp(0.0, 0.82);
            let height = 165.0 + o3_ppb * 0.95 + trap * 48.0;
            s.color = Color::srgba(0.89, 0.79, 0.18, alpha);
            s.custom_size = Some(Vec2::new(1400.0, height));
            t.translation.y = -105.0 + 0.16 * height;
        }
    }

    {
        let mut no2_q = visuals.p3();
        if let Ok((mut s, mut t)) = no2_q.get_single_mut() {
            let alpha = ((no2_ppb / 100.0) * (1.0 - 0.18 * wind_f) * trap).clamp(0.0, 0.68);
            let height = 108.0 + no2_ppb * 0.36 + trap * 18.0;
            s.color = Color::srgba(0.82, 0.33, 0.10, alpha);
            s.custom_size = Some(Vec2::new(1400.0, height));
            t.translation.y = 52.0 + 0.18 * height;
        }
    }

    {
        let mut sun_q = visuals.p4();
        if let Ok((mut t, mut s)) = sun_q.get_single_mut() {
            let frac = ((state.time_of_day - 6.0) / 12.0).clamp(0.0, 1.0);
            let angle = std::f32::consts::PI * frac;
            t.translation.x = -340.0 * angle.cos();
            t.translation.y = 300.0 * angle.sin() - 50.0;
            t.scale = Vec3::splat(
                0.82 + 0.30 * state.params.solar_flux as f32
                    + 0.02 * (state.params.temperature_c as f32 - 25.0).max(0.0),
            );
            s.color = Color::srgba(
                1.0,
                0.95,
                0.60,
                ((0.10 + 0.84 * day_f) * (1.0 - 0.26 * smog_f)).clamp(0.0, 1.0),
            );
        }
    }

    let building_light_alpha = (0.08 + 0.98 * darkness + 0.28 * smog_f).clamp(0.0, 1.0);
    let headlight_alpha = (0.02 + 1.05 * darkness + 0.50 * smog_f + 0.14 * no2_f).clamp(0.0, 1.0);
    let taillight_alpha = (0.14 + 0.26 * darkness + 0.20 * smog_f).clamp(0.10, 0.82);
    {
        let mut scene_q = visuals.p6();
        for (mut sprite, building, window, light) in scene_q.iter_mut() {
            if let Some(building) = building {
                let facade = (0.07 + 0.18 * ambient_brightness) * building.shade;
                sprite.color = Color::srgb(
                    (0.10 + 0.07 * facade).clamp(0.0, 1.0),
                    (0.11 + 0.08 * facade).clamp(0.0, 1.0),
                    (0.15 + 0.11 * facade).clamp(0.0, 1.0),
                );
                continue;
            }

            if let Some(window) = window {
                let flicker =
                    0.65 + 0.35 * (window.glow_seed + state.time_of_day * 0.35).sin().abs();
                let occupancy = 0.40 + 0.60 * (window.glow_seed * 1.37).sin().abs();
                let alpha = (building_light_alpha * flicker * occupancy).clamp(0.0, 0.98);
                sprite.color =
                    Color::srgba(1.0, 0.80 + 0.12 * flicker, 0.40 + 0.22 * flicker, alpha);
                continue;
            }

            if let Some(light) = light {
                match light.kind {
                    CarLightKind::Headlight => {
                        sprite.color = Color::srgba(1.0, 0.96, 0.84, headlight_alpha);
                    }
                    CarLightKind::Taillight => {
                        sprite.color = Color::srgba(1.0, 0.18, 0.16, taillight_alpha);
                    }
                }
            }
        }
    }

    let traffic_visual = ((traffic_now - 0.08) / 1.70).clamp(0.0, 1.0);
    let active_cars = if active_density < 0.03 {
        0usize
    } else {
        ((CAR_POOL as f32) * traffic_visual.powf(0.85))
            .round()
            .clamp(1.0, CAR_POOL as f32) as usize
    };
    let car_speed = 20.0 + 28.0 * traffic_visual + 6.0 * ambient_brightness;
    let car_alpha = (0.35 + 0.55 * traffic_visual).clamp(0.22, 1.0);
    {
        let mut car_q = visuals.p5();
        for (car, mut t, mut s, mut visibility) in car_q.iter_mut() {
            let visible = car.slot < active_cars;
            *visibility = if visible {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
            if !visible {
                continue;
            }
            t.translation.x += car.dir * car_speed * time.delta_secs();
            if t.translation.x > 760.0 {
                t.translation.x = -760.0;
            }
            if t.translation.x < -760.0 {
                t.translation.x = 760.0;
            }
            s.color = Color::srgba(
                (0.28 + 0.28 * ambient_brightness).clamp(0.0, 1.0),
                (0.08 + 0.06 * ambient_brightness).clamp(0.0, 1.0),
                (0.09 + 0.07 * ambient_brightness).clamp(0.0, 1.0),
                car_alpha,
            );
        }
    }
}

fn render_ui(
    mut ctx: EguiContexts,
    mut state: ResMut<SimState>,
    mut surrogate: ResMut<SurrogateRes>,
) {
    egui::SidePanel::right("controls")
        .min_width(360.0)
        .show(ctx.ctx_mut(), |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.heading("Photochemical Smog Demo");
                    ui.label("Expanded urban ozone chemistry with a fixed-step RK4 solver and optional ONNX surrogate.");
                    ui.separator();

                    ui.collapsing("Scene presets", |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Morning commute").clicked() {
                                apply_preset(&mut state, 7.4, 0.90, 0.85, 0.18, 23.0, 0.55, 0.18, 0.40, false);
                            }
                            if ui.button("Sunny noon").clicked() {
                                apply_preset(&mut state, 12.8, 0.55, 1.30, 0.22, 33.0, 0.32, 0.25, 0.10, false);
                            }
                            if ui.button("Industrial haze").clicked() {
                                apply_preset(&mut state, 14.0, 0.62, 1.15, 0.18, 31.0, 0.58, 0.90, 0.62, false);
                            }
                            if ui.button("Weekend clearing").clicked() {
                                apply_preset(&mut state, 11.6, 0.38, 1.05, 0.65, 27.0, 0.44, 0.18, 0.08, true);
                            }
                        });
                    });

                    ui.collapsing("Quick interventions", |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Traffic spike (1 h)").clicked() {
                                state.start_traffic_spike();
                            }
                            if ui.button("Cloud passes").clicked() {
                                state.params.solar_flux = (state.params.solar_flux - 0.18).clamp(0.2, 1.5);
                            }
                            if ui.button("Heat wave").clicked() {
                                state.params.temperature_c = (state.params.temperature_c + 3.0).clamp(10.0, 40.0);
                            }
                            if ui.button("Sea breeze").clicked() {
                                state.params.wind_speed = (state.params.wind_speed + 0.18).clamp(0.0, 1.0);
                                state.params.humidity = (state.params.humidity + 0.10).clamp(0.1, 1.0);
                            }
                            if ui.button("Inversion traps").clicked() {
                                state.params.inversion_strength = (state.params.inversion_strength + 0.18).clamp(0.0, 1.0);
                            }
                            if ui.button("Reset atmosphere").clicked() {
                                state.reset_atmosphere();
                            }
                        });
                        if state.traffic_spike_active() {
                            ui.small(format!(
                                "Traffic spike active for another {:.2} simulated hours.",
                                state.traffic_spike_remaining_hours
                            ));
                        }
                    });

                    ui.separator();
                    ui.heading("Controls");
                    let previous_time = state.time_of_day;
                    let previous_weekend_mode = state.params.weekend_mode;
                    ui.add(egui::Slider::new(&mut state.params.traffic_density, 0.0..=1.0).text("Traffic density"));
                    ui.add(egui::Slider::new(&mut state.params.industrial_emissions, 0.0..=1.0).text("Industrial emissions"));
                    ui.add(egui::Slider::new(&mut state.params.solar_flux, 0.2..=1.5).text("Solar flux / heat"));
                    ui.add(egui::Slider::new(&mut state.params.wind_speed, 0.0..=1.0).text("Wind / ventilation"));
                    ui.add(egui::Slider::new(&mut state.params.temperature_c, 10.0..=40.0).text("Temperature (°C)"));
                    ui.add(egui::Slider::new(&mut state.params.humidity, 0.1..=1.0).text("Humidity"));
                    ui.add(egui::Slider::new(&mut state.params.inversion_strength, 0.0..=1.0).text("Inversion / trapping"));
                    let weekend_response =
                        ui.checkbox(&mut state.params.weekend_mode, "Weekend traffic pattern");
                    let time_response = ui.add(
                        egui::Slider::new(&mut state.time_of_day, 0.0..=24.0)
                            .text("Time of day (h)"),
                    );
                    ui.add(egui::Slider::new(&mut state.clock_speed, 30.0..=700.0).text("Sim speed (s/s)"));
                    ui.checkbox(&mut state.auto_advance, "Auto-advance clock");
                    if ui.button(if state.paused { "Resume simulation" } else { "Pause simulation" }).clicked() {
                        state.paused = !state.paused;
                    }
                    let time_changed = time_response.changed()
                        && (state.time_of_day - previous_time).abs() > f32::EPSILON;
                    let weekend_changed =
                        weekend_response.changed() && state.params.weekend_mode != previous_weekend_mode;
                    if time_changed || weekend_changed {
                        state.reset_atmosphere();
                    }
                    ui.small("The right-hand control bar is now scrollable, so all controls remain reachable even with the expanded model.");
                    ui.small("Traffic now follows a daily cycle: very high in the morning and evening, medium around noon, and very low near midnight.");
                    ui.small("Time-of-day and weekend-pattern changes now auto-reseed the air mass so the history does not jump from an inconsistent regime.");

                    ui.separator();
                    ui.heading("Solver");
                    ui.label(format!("Chemistry step: {:.0} s", CHEM_DT));
                    let model_present = surrogate.inner.is_some();
                    ui.add_enabled_ui(model_present, |ui| {
                        ui.checkbox(&mut surrogate.enabled, "Neural surrogate (ONNX)");
                    });
                    if !model_present {
                        ui.colored_label(egui::Color32::from_rgb(210, 90, 90), "smog_surrogate.onnx not found — RK4 fallback active");
                    } else if surrogate.enabled {
                        ui.colored_label(egui::Color32::from_rgb(90, 200, 110), "NN active (10 s forecast)");
                    } else {
                        ui.label("RK4 active");
                    }

                    ui.separator();
                    ui.heading("Current chemistry");
                    ui.monospace(format!("NO2  {:6.1} ppb", state.chem.no2));
                    ui.monospace(format!("NO   {:6.1} ppb", state.chem.no));
                    ui.monospace(format!("O3   {:6.1} ppb", state.chem.o3));
                    ui.monospace(format!("VOC  {:6.1} ppb", state.chem.voc));
                    let smog_label = smog_index_label(state.chem.o3);
                    ui.colored_label(smog_label.1, format!("Smog index: {}", smog_label.0));
                    ui.label(format!("Regime: {}", chemistry_phase(&state)));
                    ui.label(format!("Narrative: {}", scenario_narrative(&state)));

                    ui.separator();
                    ui.heading("Drivers");
                    let hour = state.time_of_day as f64;
                    let active_density = state.active_traffic_density();
                    let traffic_now = traffic_profile(hour, active_density, state.params.weekend_mode);
                    let live_params = state.effective_params();
                    let mix_now = mixing_coeff(hour, state.params.solar_flux, state.params.wind_speed, state.params.inversion_strength);
                    let daylight = solar_arc(hour) * state.params.solar_flux;
                    let temp_factor = temperature_factor(state.params.temperature_c);
                    let humidity_factor = humidity_factor(state.params.humidity);
                    let trap_factor = trapping_factor(state.params.inversion_strength, state.params.wind_speed);
                    ui.monospace(format!("Daylight factor      {:5.2}", daylight));
                    ui.monospace(format!("Photolysis j1        {:5.4} s^-1", j1(hour, state.params.solar_flux)));
                    ui.monospace(format!("Base traffic density {:5.2}", state.params.traffic_density));
                    ui.monospace(format!("Live traffic density {:5.2}", live_params.traffic_density));
                    ui.monospace(format!("Traffic activity     {:5.2}", traffic_now));
                    ui.monospace(format!("Mixing coeff         {:5.5} s^-1", mix_now));
                    ui.monospace(format!("Temp chemistry boost {:5.2}", temp_factor));
                    ui.monospace(format!("Humidity factor      {:5.2}", humidity_factor));
                    ui.monospace(format!("Trap factor          {:5.2}", trap_factor));

                    ui.separator();
                    ui.heading("Recent concentration history");
                    draw_history_plot(ui, &state.history);

                    ui.separator();
                    ui.collapsing("How to read the demo", |ui| {
                        ui.label("Traffic emits fresh NOx, industry feeds VOCs, sunlight photolyzes NO2, and warm conditions accelerate ozone production. Wind mixes the box, while inversions trap pollutants near the skyline. Humidity can both support radical chemistry and increase ozone loss, so visitors can explore competing effects.");
                    });
                });
        });
}

fn apply_preset(
    state: &mut SimState,
    hour: f32,
    traffic: f64,
    solar: f64,
    wind: f64,
    temp_c: f64,
    humidity: f64,
    industry: f64,
    inversion: f64,
    weekend: bool,
) {
    state.time_of_day = hour;
    state.params.traffic_density = traffic;
    state.params.solar_flux = solar;
    state.params.wind_speed = wind;
    state.params.temperature_c = temp_c;
    state.params.humidity = humidity;
    state.params.industrial_emissions = industry;
    state.params.inversion_strength = inversion;
    state.params.weekend_mode = weekend;
    state.clear_interventions();
    state.reset_atmosphere();
}

fn smog_index_label(o3: f64) -> (&'static str, egui::Color32) {
    match o3 as u32 {
        0..=39 => ("Good", egui::Color32::from_rgb(90, 200, 90)),
        40..=69 => ("Watch", egui::Color32::from_rgb(210, 200, 70)),
        70..=99 => ("Smog building", egui::Color32::from_rgb(235, 150, 40)),
        100..=139 => (
            "Unhealthy (sensitive)",
            egui::Color32::from_rgb(230, 105, 40),
        ),
        _ => ("Unhealthy", egui::Color32::from_rgb(205, 55, 55)),
    }
}

fn chemistry_phase(state: &SimState) -> &'static str {
    let hour = state.time_of_day as f64;
    let daylight = solar_arc(hour) * state.params.solar_flux;
    if daylight < 0.05 {
        "Nighttime titration / residual-layer storage"
    } else if state.params.inversion_strength > 0.65 && state.params.wind_speed < 0.25 {
        "Trapped urban plume under inversion"
    } else if state.chem.no > 35.0 && state.chem.o3 < 18.0 {
        "Fresh traffic plume suppressing ozone"
    } else if state.params.wind_speed > 0.75 {
        "Ventilated atmosphere clearing the haze"
    } else if state.chem.o3 > 85.0 {
        "Photochemical smog episode"
    } else {
        "Ozone building under sunlight"
    }
}

fn scenario_narrative(state: &SimState) -> &'static str {
    if state.params.weekend_mode && state.params.industrial_emissions > 0.55 {
        "Lower traffic but persistent industrial VOC loading."
    } else if state.params.temperature_c > 32.0 && state.params.solar_flux > 1.1 {
        "Hot bright conditions favor fast ozone buildup."
    } else if state.params.inversion_strength > 0.55 {
        "The boundary layer is trapping pollutants near street level."
    } else if state.params.wind_speed > 0.6 {
        "Ventilation is diluting the urban plume."
    } else {
        "Balanced urban background with evolving sunlight chemistry."
    }
}

fn draw_history_plot(ui: &mut egui::Ui, history: &VecDeque<HistoryPoint>) {
    let desired_size = egui::vec2(ui.available_width(), 160.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter();

    if history.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "History fills as the chemistry advances",
            egui::TextStyle::Body.resolve(ui.style()),
            egui::Color32::GRAY,
        );
        return;
    }

    let max_y = history.iter().fold(120.0_f32, |acc, p| {
        acc.max(p.o3).max(p.no2).max(p.no).max(p.voc * 0.35)
    });

    let to_pos = |i: usize, value: f32| {
        let x = rect.left() + rect.width() * (i as f32 / (history.len() - 1) as f32);
        let y = rect.bottom() - rect.height() * (value / max_y).clamp(0.0, 1.0);
        egui::pos2(x, y)
    };

    painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));

    for (series, color) in [
        (
            history
                .iter()
                .enumerate()
                .map(|(i, p)| to_pos(i, p.o3))
                .collect::<Vec<_>>(),
            egui::Color32::from_rgb(220, 185, 60),
        ),
        (
            history
                .iter()
                .enumerate()
                .map(|(i, p)| to_pos(i, p.no2))
                .collect::<Vec<_>>(),
            egui::Color32::from_rgb(210, 100, 55),
        ),
        (
            history
                .iter()
                .enumerate()
                .map(|(i, p)| to_pos(i, p.no))
                .collect::<Vec<_>>(),
            egui::Color32::from_rgb(120, 205, 255),
        ),
        (
            history
                .iter()
                .enumerate()
                .map(|(i, p)| to_pos(i, p.voc * 0.35))
                .collect::<Vec<_>>(),
            egui::Color32::from_rgb(155, 110, 210),
        ),
    ] {
        painter.add(egui::Shape::line(series, egui::Stroke::new(2.0, color)));
    }

    painter.text(
        rect.left_top() + egui::vec2(8.0, 6.0),
        egui::Align2::LEFT_TOP,
        "O3",
        egui::TextStyle::Small.resolve(ui.style()),
        egui::Color32::from_rgb(220, 185, 60),
    );
    painter.text(
        rect.left_top() + egui::vec2(38.0, 6.0),
        egui::Align2::LEFT_TOP,
        "NO2",
        egui::TextStyle::Small.resolve(ui.style()),
        egui::Color32::from_rgb(210, 100, 55),
    );
    painter.text(
        rect.left_top() + egui::vec2(78.0, 6.0),
        egui::Align2::LEFT_TOP,
        "NO",
        egui::TextStyle::Small.resolve(ui.style()),
        egui::Color32::from_rgb(120, 205, 255),
    );
    painter.text(
        rect.left_top() + egui::vec2(108.0, 6.0),
        egui::Align2::LEFT_TOP,
        "VOC x0.35",
        egui::TextStyle::Small.resolve(ui.style()),
        egui::Color32::from_rgb(155, 110, 210),
    );
}
