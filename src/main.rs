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
const ROUND_SECONDS: f32 = 90.0;
const POLICY_TIMES: [f32; 3] = [20.0, 50.0, 70.0];
const CARD_COUNT: usize = 10;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "90-Second Smog Cards - AI-Lab x ChemiClub".into(),
                resolution: (1440.0, 820.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .init_resource::<SimState>()
        .init_resource::<MechanismImage>()
        .insert_resource(SurrogateRes::default())
        .add_systems(Startup, (load_surrogate, setup_scene))
        .add_systems(Update, (advance_chemistry, sync_visuals, render_ui).chain())
        .run();
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UiMode {
    Game,
    Lab,
}

#[derive(Clone, Copy)]
struct HistoryPoint {
    o3: f32,
    no2: f32,
    no: f32,
    voc: f32,
}

#[derive(Resource)]
struct MechanismImage {
    handle: Handle<Image>,
}

impl FromWorld for MechanismImage {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        Self {
            handle: asset_server.load("smog_mechanism.png"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CardId {
    TrafficLimit,
    FactoryPause,
    VocScrubber,
    SunlightShield,
    VentilationCorridor,
    HealthAdvisory,
    RushHourPriority,
    FactoryOvertime,
    HeatingSubsidy,
    FestivalApproval,
}

impl CardId {
    fn idx(self) -> usize {
        match self {
            CardId::TrafficLimit => 0,
            CardId::FactoryPause => 1,
            CardId::VocScrubber => 2,
            CardId::SunlightShield => 3,
            CardId::VentilationCorridor => 4,
            CardId::HealthAdvisory => 5,
            CardId::RushHourPriority => 6,
            CardId::FactoryOvertime => 7,
            CardId::HeatingSubsidy => 8,
            CardId::FestivalApproval => 9,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CardKind {
    Active,
    Instant,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CardIntent {
    Air,
    City,
    Health,
}

#[derive(Clone, Copy)]
struct CardDef {
    name: &'static str,
    target: &'static str,
    effect: &'static str,
    kind: CardKind,
    intent: CardIntent,
    duration: f32,
    cooldown: f32,
    city_delta: f32,
}

fn card_def(card: CardId) -> CardDef {
    match card {
        CardId::TrafficLimit => CardDef {
            name: "Traffic Limit",
            target: "NOx",
            effect: "NOx emissions -35%",
            kind: CardKind::Active,
            intent: CardIntent::Air,
            duration: 22.0,
            cooldown: 28.0,
            city_delta: -12.0,
        },
        CardId::FactoryPause => CardDef {
            name: "Factory Pause",
            target: "VOC",
            effect: "VOC source -55%",
            kind: CardKind::Instant,
            intent: CardIntent::Air,
            duration: 12.0,
            cooldown: 24.0,
            city_delta: -24.0,
        },
        CardId::VocScrubber => CardDef {
            name: "VOC Scrubber",
            target: "VOC",
            effect: "VOC emissions -35%",
            kind: CardKind::Active,
            intent: CardIntent::Air,
            duration: 25.0,
            cooldown: 31.0,
            city_delta: -13.0,
        },
        CardId::SunlightShield => CardDef {
            name: "Sunlight Shield",
            target: "Sun",
            effect: "photolysis -28%",
            kind: CardKind::Active,
            intent: CardIntent::Air,
            duration: 18.0,
            cooldown: 25.0,
            city_delta: -11.0,
        },
        CardId::VentilationCorridor => CardDef {
            name: "Ventilation Corridor",
            target: "Mixing",
            effect: "mixing +40%",
            kind: CardKind::Active,
            intent: CardIntent::Air,
            duration: 24.0,
            cooldown: 30.0,
            city_delta: -10.0,
        },
        CardId::HealthAdvisory => CardDef {
            name: "Health Advisory",
            target: "Health",
            effect: "exposure damage -18%",
            kind: CardKind::Instant,
            intent: CardIntent::Health,
            duration: 20.0,
            cooldown: 30.0,
            city_delta: -5.0,
        },
        CardId::RushHourPriority => CardDef {
            name: "Rush-Hour Priority",
            target: "City",
            effect: "stability +12, NOx +25%",
            kind: CardKind::Active,
            intent: CardIntent::City,
            duration: 20.0,
            cooldown: 26.0,
            city_delta: 12.0,
        },
        CardId::FactoryOvertime => CardDef {
            name: "Factory Overtime",
            target: "City",
            effect: "stability +15, VOC +30%",
            kind: CardKind::Active,
            intent: CardIntent::City,
            duration: 22.0,
            cooldown: 28.0,
            city_delta: 15.0,
        },
        CardId::HeatingSubsidy => CardDef {
            name: "Heating Subsidy",
            target: "City",
            effect: "stability +10, NOx +22%",
            kind: CardKind::Active,
            intent: CardIntent::City,
            duration: 24.0,
            cooldown: 30.0,
            city_delta: 10.0,
        },
        CardId::FestivalApproval => CardDef {
            name: "Festival Approval",
            target: "City",
            effect: "stability +18, haze spike",
            kind: CardKind::Instant,
            intent: CardIntent::City,
            duration: 12.0,
            cooldown: 30.0,
            city_delta: 18.0,
        },
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IncidentId {
    HeatWave,
    TrafficJam,
    FactoryLeak,
    ColdSnap,
    TemperatureInversion,
    CloudBreak,
    FestivalFireworks,
    WindShift,
}

#[derive(Clone, Copy)]
struct IncidentDef {
    name: &'static str,
    effect: &'static str,
    counter: &'static str,
    duration: f32,
}

fn incident_def(id: IncidentId) -> IncidentDef {
    match id {
        IncidentId::HeatWave => IncidentDef {
            name: "Heat Wave",
            effect: "Sunlight and temperature jump.",
            counter: "Counter: Sun or VOC card",
            duration: 24.0,
        },
        IncidentId::TrafficJam => IncidentDef {
            name: "Traffic Jam",
            effect: "Vehicle NOx surges.",
            counter: "Counter: NOx policy",
            duration: 22.0,
        },
        IncidentId::FactoryLeak => IncidentDef {
            name: "Factory Solvent Leak",
            effect: "VOC source spikes.",
            counter: "Counter: VOC policy",
            duration: 24.0,
        },
        IncidentId::ColdSnap => IncidentDef {
            name: "Cold Snap",
            effect: "Heating combustion raises NOx.",
            counter: "Counter: NOx or mixing",
            duration: 25.0,
        },
        IncidentId::TemperatureInversion => IncidentDef {
            name: "Temperature Inversion",
            effect: "Mixing collapses; smog is trapped.",
            counter: "Counter: ventilation",
            duration: 26.0,
        },
        IncidentId::CloudBreak => IncidentDef {
            name: "Cloud Break",
            effect: "Stored precursors meet strong sun.",
            counter: "Counter: Sun or VOC card",
            duration: 14.0,
        },
        IncidentId::FestivalFireworks => IncidentDef {
            name: "Festival Fireworks",
            effect: "Short NOx and haze pulse.",
            counter: "Counter: health or mixing",
            duration: 16.0,
        },
        IncidentId::WindShift => IncidentDef {
            name: "Wind Shift",
            effect: "Ventilation weakens.",
            counter: "Counter: ventilation",
            duration: 24.0,
        },
    }
}

#[derive(Clone, Copy)]
struct ActiveEffect {
    card: CardId,
    remaining: f32,
}

#[derive(Clone, Copy)]
struct ScheduledIncident {
    id: IncidentId,
    start: f32,
    duration: f32,
    triggered: bool,
}

impl ScheduledIncident {
    fn is_active(&self, elapsed: f32) -> bool {
        self.triggered && elapsed <= self.start + self.duration
    }
}

#[derive(Clone, Copy)]
struct PolicyChoice {
    replace_index: Option<usize>,
    offers: [CardId; 2],
}

struct GameState {
    round_elapsed: f32,
    city_stability: f32,
    air_safety_sum: f32,
    air_safety_time: f32,
    round_over: bool,
    final_score: Option<f32>,
    hand: [CardId; 3],
    active_slots: [Option<ActiveEffect>; 2],
    temp_effects: Vec<ActiveEffect>,
    cooldowns: [f32; CARD_COUNT],
    scheduled_incidents: Vec<ScheduledIncident>,
    next_policy_index: usize,
    policy_choice: Option<PolicyChoice>,
    rng: u64,
    message: String,
}

impl Default for GameState {
    fn default() -> Self {
        let mut game = Self {
            round_elapsed: 0.0,
            city_stability: 75.0,
            air_safety_sum: 0.0,
            air_safety_time: 0.0,
            round_over: false,
            final_score: None,
            hand: [
                CardId::TrafficLimit,
                CardId::VocScrubber,
                CardId::RushHourPriority,
            ],
            active_slots: [None, None],
            temp_effects: Vec::new(),
            cooldowns: [0.0; CARD_COUNT],
            scheduled_incidents: Vec::new(),
            next_policy_index: 0,
            policy_choice: None,
            rng: 0x5EED_2026_CAFE_BABE,
            message: "Keep Air Safety and City Stability balanced.".to_string(),
        };
        game.schedule_incidents();
        game
    }
}

impl GameState {
    fn reset(&mut self) {
        self.round_elapsed = 0.0;
        self.city_stability = 75.0;
        self.air_safety_sum = 0.0;
        self.air_safety_time = 0.0;
        self.round_over = false;
        self.final_score = None;
        self.hand = [
            CardId::TrafficLimit,
            CardId::VocScrubber,
            CardId::RushHourPriority,
        ];
        self.active_slots = [None, None];
        self.temp_effects.clear();
        self.cooldowns = [0.0; CARD_COUNT];
        self.next_policy_index = 0;
        self.policy_choice = None;
        self.message = "New round: balance clean air against city pressure.".to_string();
        self.schedule_incidents();
    }

    fn next_rand(&mut self) -> u32 {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng >> 32) as u32
    }

    fn rand_index(&mut self, len: usize) -> usize {
        (self.next_rand() as usize) % len
    }

    fn schedule_incidents(&mut self) {
        const INCIDENTS: [IncidentId; 8] = [
            IncidentId::HeatWave,
            IncidentId::TrafficJam,
            IncidentId::FactoryLeak,
            IncidentId::ColdSnap,
            IncidentId::TemperatureInversion,
            IncidentId::CloudBreak,
            IncidentId::FestivalFireworks,
            IncidentId::WindShift,
        ];

        self.scheduled_incidents.clear();
        let first = INCIDENTS[self.rand_index(INCIDENTS.len())];
        self.scheduled_incidents.push(ScheduledIncident {
            id: first,
            start: 12.0,
            duration: incident_def(first).duration,
            triggered: false,
        });

        if self.next_rand() % 100 < 75 {
            let mut second = INCIDENTS[self.rand_index(INCIDENTS.len())];
            if second == first {
                second = INCIDENTS[(self.rand_index(INCIDENTS.len() - 1) + 1) % INCIDENTS.len()];
            }
            self.scheduled_incidents.push(ScheduledIncident {
                id: second,
                start: 45.0,
                duration: incident_def(second).duration,
                triggered: false,
            });
        }
    }

    fn next_policy_pair(&mut self) -> [CardId; 2] {
        const PAIRS: [[CardId; 2]; 5] = [
            [CardId::TrafficLimit, CardId::RushHourPriority],
            [CardId::VocScrubber, CardId::FactoryOvertime],
            [CardId::SunlightShield, CardId::FestivalApproval],
            [CardId::VentilationCorridor, CardId::HeatingSubsidy],
            [CardId::FactoryPause, CardId::HealthAdvisory],
        ];
        PAIRS[self.rand_index(PAIRS.len())]
    }

    fn average_air_safety(&self) -> f32 {
        if self.air_safety_time <= 0.0 {
            100.0
        } else {
            self.air_safety_sum / self.air_safety_time
        }
    }
}

#[derive(Resource)]
struct SimState {
    chem: ChemState,
    params: SmogParams,
    time_of_day: f32,
    auto_advance: bool,
    paused: bool,
    clock_speed: f32,
    step_accumulator: f64,
    history: VecDeque<HistoryPoint>,
    mode: UiMode,
    game: GameState,
}

impl Default for SimState {
    fn default() -> Self {
        let params = SmogParams::default();
        let time_of_day = 7.2;
        let chem = ChemState::urban_baseline(time_of_day as f64, &params);
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
            clock_speed: 420.0,
            step_accumulator: 0.0,
            history,
            mode: UiMode::Game,
            game: GameState::default(),
        }
    }
}

impl SimState {
    fn reset_atmosphere(&mut self) {
        self.chem = ChemState::urban_baseline(self.time_of_day as f64, &self.params);
        self.step_accumulator = 0.0;
        self.history.clear();
        self.push_history_point();
    }

    fn reset_game_round(&mut self) {
        self.mode = UiMode::Game;
        self.params = SmogParams::default();
        self.time_of_day = 7.2;
        self.auto_advance = true;
        self.paused = false;
        self.clock_speed = 420.0;
        self.game.reset();
        self.reset_atmosphere();
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

    fn effective_params(&self) -> SmogParams {
        let mut params = self.params.clone();
        if self.mode == UiMode::Game {
            for slot in &self.game.active_slots {
                if let Some(effect) = slot {
                    apply_card_effect(effect.card, &mut params);
                }
            }
            for effect in &self.game.temp_effects {
                apply_card_effect(effect.card, &mut params);
            }
            for incident in &self.game.scheduled_incidents {
                if incident.is_active(self.game.round_elapsed) {
                    apply_incident_effect(incident.id, &mut params);
                }
            }
        }
        params
    }

    fn active_incident_ids(&self) -> Vec<IncidentId> {
        if self.mode != UiMode::Game {
            return Vec::new();
        }
        self.game
            .scheduled_incidents
            .iter()
            .filter(|incident| incident.is_active(self.game.round_elapsed))
            .map(|incident| incident.id)
            .collect()
    }

    fn is_incident_active(&self, id: IncidentId) -> bool {
        self.active_incident_ids().contains(&id)
    }

    fn is_card_running(&self, card: CardId) -> bool {
        self.game
            .active_slots
            .iter()
            .flatten()
            .any(|effect| effect.card == card)
            || self
                .game
                .temp_effects
                .iter()
                .any(|effect| effect.card == card)
    }

    fn protection_factor(&self) -> f32 {
        if self.is_card_running(CardId::HealthAdvisory) {
            0.18
        } else {
            0.0
        }
    }

    fn current_air_safety(&self) -> f32 {
        let params = self.effective_params();
        air_safety_index(
            &self.chem,
            &params,
            self.time_of_day as f64,
            self.protection_factor(),
        )
    }

    fn tick_game(&mut self, dt: f32) {
        if self.mode != UiMode::Game || self.game.round_over {
            return;
        }

        self.game.round_elapsed = (self.game.round_elapsed + dt).min(ROUND_SECONDS);

        for slot in &mut self.game.active_slots {
            if let Some(effect) = slot.as_mut() {
                effect.remaining -= dt;
            }
            if slot.as_ref().is_some_and(|effect| effect.remaining <= 0.0) {
                *slot = None;
            }
        }

        for effect in &mut self.game.temp_effects {
            effect.remaining -= dt;
        }
        self.game
            .temp_effects
            .retain(|effect| effect.remaining > 0.0);

        for cooldown in &mut self.game.cooldowns {
            *cooldown = (*cooldown - dt).max(0.0);
        }

        for incident in &mut self.game.scheduled_incidents {
            if !incident.triggered && self.game.round_elapsed >= incident.start {
                incident.triggered = true;
                let def = incident_def(incident.id);
                self.game.message = format!("Incident: {}. {}", def.name, def.counter);
            }
        }

        if self.game.next_policy_index < POLICY_TIMES.len()
            && self.game.round_elapsed >= POLICY_TIMES[self.game.next_policy_index]
            && self.game.policy_choice.is_none()
        {
            let offers = self.game.next_policy_pair();
            self.game.policy_choice = Some(PolicyChoice {
                replace_index: None,
                offers,
            });
            self.game.next_policy_index += 1;
            self.game.message = "Policy choice opened: replace one card.".to_string();
        }

        let safety = self.current_air_safety();
        self.game.air_safety_sum += safety * dt;
        self.game.air_safety_time += dt;

        if self.game.round_elapsed >= ROUND_SECONDS {
            let avg_safety = self.game.average_air_safety();
            let final_score = avg_safety.min(self.game.city_stability);
            self.game.final_score = Some(final_score);
            self.game.round_over = true;
            self.game.message = result_reason(avg_safety, self.game.city_stability).to_string();
        }
    }

    fn can_play_hand_card(&self, hand_index: usize) -> bool {
        if self.mode != UiMode::Game || self.game.round_over || hand_index >= self.game.hand.len() {
            return false;
        }
        let card = self.game.hand[hand_index];
        let def = card_def(card);
        if self.game.cooldowns[card.idx()] > 0.0 || self.is_card_running(card) {
            return false;
        }
        if def.kind == CardKind::Active && self.game.active_slots.iter().all(Option::is_some) {
            return false;
        }
        true
    }

    fn play_hand_card(&mut self, hand_index: usize) {
        if !self.can_play_hand_card(hand_index) {
            self.game.message = "That policy cannot be played right now.".to_string();
            return;
        }

        let card = self.game.hand[hand_index];
        let def = card_def(card);
        let effect = ActiveEffect {
            card,
            remaining: def.duration,
        };

        match def.kind {
            CardKind::Active => {
                if let Some(slot) = self
                    .game
                    .active_slots
                    .iter_mut()
                    .find(|slot| slot.is_none())
                {
                    *slot = Some(effect);
                }
            }
            CardKind::Instant => {
                self.game.temp_effects.push(effect);
            }
        }

        self.game.city_stability = (self.game.city_stability + def.city_delta).clamp(0.0, 100.0);
        self.game.cooldowns[card.idx()] = def.cooldown;
        self.game.message = if def.city_delta >= 0.0 {
            format!("{} boosts stability but worsens chemistry.", def.name)
        } else {
            format!("{} protects air at a city cost.", def.name)
        };
    }

    fn accept_policy_offer(&mut self, offer_index: usize) {
        if let Some(choice) = self.game.policy_choice {
            if let Some(replace_index) = choice.replace_index {
                self.game.hand[replace_index] = choice.offers[offer_index.min(1)];
                self.game.policy_choice = None;
                self.game.message = "Policy hand updated.".to_string();
            }
        }
    }
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
struct IncidentPlume;
#[derive(Component)]
struct InversionCap;
#[derive(Component)]
struct CarMarker {
    dir: f32,
}

fn load_surrogate(mut res: ResMut<SurrogateRes>) {
    use surrogate::NNSurrogate;
    match NNSurrogate::load("smog_surrogate.onnx") {
        Some(nn) => {
            res.inner = Some(nn);
            res.enabled = true;
            println!("[surrogate] smog_surrogate.onnx loaded - neural solver active");
        }
        None => {
            println!("[surrogate] smog_surrogate.onnx not found - falling back to RK4");
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
            color: Color::srgba(0.55, 0.50, 0.58, 0.0),
            custom_size: Some(Vec2::new(190.0, 280.0)),
            ..default()
        },
        Transform::from_xyz(500.0, -95.0, 1.4),
        IncidentPlume,
    ));

    commands.spawn((
        Sprite {
            color: Color::srgba(0.72, 0.72, 0.76, 0.0),
            custom_size: Some(Vec2::new(1450.0, 42.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 170.0, 1.5),
        InversionCap,
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
    for &(x, h, w) in buildings {
        commands.spawn((
            Sprite {
                color: Color::srgb(0.17, 0.18, 0.22),
                custom_size: Some(Vec2::new(w, h)),
                ..default()
            },
            Transform::from_xyz(x, -285.0 + h * 0.5, 3.0),
        ));
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

    for i in 0..12usize {
        let x = -700.0 + i as f32 * 120.0;
        let dir = if i % 2 == 0 { 1.0_f32 } else { -1.0 };
        let y = if dir > 0.0 { -298.0 } else { -315.0 };
        commands.spawn((
            Sprite {
                color: Color::srgba(0.62, 0.13, 0.12, 0.80),
                custom_size: Some(Vec2::new(44.0, 16.0)),
                ..default()
            },
            Transform::from_xyz(x, y, 4.0),
            CarMarker { dir },
        ));
    }
}

fn advance_chemistry(
    mut state: ResMut<SimState>,
    mut surrogate: ResMut<SurrogateRes>,
    time: Res<Time>,
) {
    if !state.paused {
        state.tick_game(time.delta_secs());
    }

    if state.paused || (state.mode == UiMode::Game && state.game.round_over) {
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
        Query<(&CarMarker, &mut Transform, &mut Sprite)>,
        Query<&mut Sprite, With<IncidentPlume>>,
        Query<&mut Sprite, With<InversionCap>>,
    )>,
) {
    let params = state.effective_params();
    let o3_ppb = (state.chem.o3 as f32).clamp(0.0, 260.0);
    let no2_ppb = (state.chem.no2 as f32).clamp(0.0, 220.0);
    let hour = state.time_of_day as f64;
    let day_f = solar_arc(hour) as f32 * params.solar_flux as f32;
    let wind_f = params.wind_speed as f32;
    let traffic_now = traffic_profile(hour, params.traffic_density, params.weekend_mode) as f32;
    let trap = trapping_factor(params.inversion_strength, params.wind_speed) as f32;
    let humidity = params.humidity as f32;
    let heat_wave = state.is_incident_active(IncidentId::HeatWave);
    let cold_snap = state.is_incident_active(IncidentId::ColdSnap);
    let factory_plume = state.is_incident_active(IncidentId::FactoryLeak)
        || state.is_card_running(CardId::FactoryOvertime);
    let inversion = state.is_incident_active(IncidentId::TemperatureInversion)
        || state.is_incident_active(IncidentId::WindShift)
        || params.inversion_strength > 0.55;

    {
        let mut sky_q = visuals.p0();
        if let Ok(mut s) = sky_q.get_single_mut() {
            let smog_f = (o3_ppb / 155.0).clamp(0.0, 1.0);
            let humid_tint = 0.08 * humidity;
            let heat_tint = if heat_wave { 0.10 } else { 0.0 };
            let cold_tint = if cold_snap { 0.12 } else { 0.0 };
            s.color = Color::srgb(
                0.10 + 0.30 * day_f + 0.20 * smog_f + heat_tint - 0.05 * cold_tint,
                0.16 + 0.52 * day_f - 0.14 * smog_f + humid_tint - 0.05 * heat_tint,
                0.24 + 0.56 * day_f - 0.25 * smog_f + humid_tint + cold_tint,
            );
        }
    }

    {
        let mut cloud_q = visuals.p1();
        if let Ok(mut c) = cloud_q.get_single_mut() {
            let cloudiness =
                (((1.12 - params.solar_flux as f32) / 0.92) + 0.25 * humidity).clamp(0.0, 1.0);
            c.color = Color::srgba(0.96, 0.97, 0.99, 0.06 + 0.36 * cloudiness);
        }
    }

    {
        let mut smog_q = visuals.p2();
        if let Ok((mut s, mut t)) = smog_q.get_single_mut() {
            let alpha =
                ((o3_ppb / 135.0) * (1.0 - 0.28 * wind_f) * (0.75 + 0.25 * trap)).clamp(0.0, 0.82);
            let height = 165.0 + o3_ppb * 0.95 + trap * 48.0;
            let purple = if factory_plume { 0.12 } else { 0.0 };
            s.color = Color::srgba(0.89 - purple, 0.79 - purple * 0.6, 0.18 + purple, alpha);
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
                0.82 + 0.30 * params.solar_flux as f32
                    + 0.02 * (params.temperature_c as f32 - 25.0).max(0.0),
            );
            s.color = Color::srgba(1.0, 0.95, 0.60, (0.16 + 0.84 * day_f).clamp(0.0, 1.0));
        }
    }

    let car_speed = 28.0 + 34.0 * traffic_now;
    let car_alpha =
        (0.15 + 0.18 * params.traffic_density as f32 + 0.32 * traffic_now).clamp(0.18, 1.0);
    {
        let mut car_q = visuals.p5();
        for (car, mut t, mut s) in car_q.iter_mut() {
            t.translation.x += car.dir * car_speed * time.delta_secs();
            if t.translation.x > 760.0 {
                t.translation.x = -760.0;
            }
            if t.translation.x < -760.0 {
                t.translation.x = 760.0;
            }
            s.color = Color::srgba(0.62, 0.13, 0.12, car_alpha);
        }
    }

    {
        let mut plume_q = visuals.p6();
        if let Ok(mut plume) = plume_q.get_single_mut() {
            let alpha = if factory_plume {
                (0.22 + state.chem.voc as f32 / 900.0).clamp(0.0, 0.58)
            } else {
                0.0
            };
            plume.color = Color::srgba(0.46, 0.35, 0.62, alpha);
        }
    }

    {
        let mut cap_q = visuals.p7();
        if let Ok(mut cap) = cap_q.get_single_mut() {
            let alpha = if inversion {
                (0.16 + params.inversion_strength as f32 * 0.32).clamp(0.0, 0.55)
            } else {
                0.0
            };
            cap.color = Color::srgba(0.72, 0.72, 0.76, alpha);
        }
    }
}

fn render_ui(
    mut ctx: EguiContexts,
    mut state: ResMut<SimState>,
    mut surrogate: ResMut<SurrogateRes>,
    mechanism_image: Res<MechanismImage>,
    mut mechanism_texture: Local<Option<egui::TextureId>>,
) {
    let mechanism_texture = *mechanism_texture
        .get_or_insert_with(|| ctx.add_image(mechanism_image.handle.clone_weak()));

    if state.mode == UiMode::Game {
        render_game_ui(ctx.ctx_mut(), &mut state, mechanism_texture);
        return;
    }

    render_lab_ui(ctx.ctx_mut(), &mut state, &mut surrogate);
}

fn render_game_ui(ctx: &egui::Context, state: &mut SimState, mechanism_texture: egui::TextureId) {
    let current_safety = state.current_air_safety();
    let avg_safety = state.game.average_air_safety();
    let final_balance = avg_safety.min(state.game.city_stability);
    let seconds_left = (ROUND_SECONDS - state.game.round_elapsed).max(0.0);

    egui::TopBottomPanel::top("game_top_bar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.heading("90-Second Smog Cards");
            ui.separator();
            ui.label(format!("{:02.0}s left", seconds_left.ceil()));
            ui.separator();
            meter(
                ui,
                "Air Safety",
                current_safety,
                egui::Color32::from_rgb(80, 190, 115),
            );
            meter(
                ui,
                "City Stability",
                state.game.city_stability,
                egui::Color32::from_rgb(90, 150, 230),
            );
            meter(
                ui,
                "Balance",
                final_balance,
                egui::Color32::from_rgb(215, 185, 75),
            );
        });
    });

    egui::SidePanel::right("game_status")
        .min_width(350.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Restart round").clicked() {
                            state.reset_game_round();
                        }
                        if ui.button("Lab mode").clicked() {
                            state.mode = UiMode::Lab;
                        }
                        ui.checkbox(&mut state.paused, "Pause");
                    });
                    ui.separator();
                    ui.heading("Round Status");
                    ui.label(&state.game.message);
                    if state.game.round_over {
                        ui.separator();
                        ui.heading(format!("Grade: {}", grade_label(final_balance)));
                        ui.label(format!("Average Air Safety: {:.0}", avg_safety));
                        ui.label(format!("City Stability: {:.0}", state.game.city_stability));
                        ui.label(format!("Final Balance: {:.0}", final_balance));
                    }

                    ui.separator();
                    ui.heading("Active Incident");
                    let incidents = state.active_incident_ids();
                    if incidents.is_empty() {
                        ui.label(next_incident_label(state));
                    } else {
                        for id in incidents {
                            let def = incident_def(id);
                            ui.colored_label(egui::Color32::from_rgb(235, 145, 70), def.name);
                            ui.label(def.effect);
                            ui.label(def.counter);
                        }
                    }

                    ui.separator();
                    ui.heading("Chemistry");
                    let params = state.effective_params();
                    let nox = state.chem.no + state.chem.no2;
                    ui.monospace(format!("O3    {:6.1} ppb", state.chem.o3));
                    ui.monospace(format!("NOx   {:6.1} ppb", nox));
                    ui.monospace(format!("VOC   {:6.1} ppb", state.chem.voc));
                    ui.monospace(format!(
                        "Sun   {:6.2}",
                        solar_arc(state.time_of_day as f64) * params.solar_flux
                    ));
                    ui.monospace(format!(
                        "Mix   {:6.4}",
                        mixing_coeff(
                            state.time_of_day as f64,
                            params.solar_flux,
                            params.wind_speed,
                            params.inversion_strength,
                        )
                    ));
                    ui.label(format!("Reaction: {}", active_reaction_label(state)));
                    render_mechanism_panel(ui, state, mechanism_texture);

                    ui.separator();
                    ui.heading("Running Policies");
                    for (i, slot) in state.game.active_slots.iter().enumerate() {
                        if let Some(effect) = slot {
                            let def = card_def(effect.card);
                            ui.label(format!(
                                "Slot {}: {} ({:.0}s)",
                                i + 1,
                                def.name,
                                effect.remaining.ceil()
                            ));
                        } else {
                            ui.label(format!("Slot {}: empty", i + 1));
                        }
                    }
                    for effect in &state.game.temp_effects {
                        let def = card_def(effect.card);
                        ui.label(format!(
                            "Timed: {} ({:.0}s)",
                            def.name,
                            effect.remaining.ceil()
                        ));
                    }
                });
        });

    egui::TopBottomPanel::bottom("card_hand")
        .resizable(false)
        .exact_height(206.0)
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("Policy Cards");
                ui.label(
                    "Two active slots. City-first cards recover stability but worsen chemistry.",
                );
            });
            ui.add_space(4.0);
            ui.horizontal_top(|ui| {
                let card_width =
                    ((ui.available_width() - 16.0) / state.game.hand.len() as f32).max(220.0);
                for hand_index in 0..state.game.hand.len() {
                    render_card_button(ui, state, hand_index, card_width);
                }
            });
        });

    render_policy_choice(ctx, state);
}

fn render_card_button(ui: &mut egui::Ui, state: &mut SimState, hand_index: usize, width: f32) {
    let card = state.game.hand[hand_index];
    let def = card_def(card);
    let cooldown = state.game.cooldowns[card.idx()];
    let can_play = state.can_play_hand_card(hand_index);
    let fill = match def.intent {
        CardIntent::Air => egui::Color32::from_rgb(29, 70, 52),
        CardIntent::City => egui::Color32::from_rgb(86, 55, 26),
        CardIntent::Health => egui::Color32::from_rgb(36, 55, 82),
    };

    egui::Frame::group(ui.style()).fill(fill).show(ui, |ui| {
        ui.set_min_width(width);
        ui.set_max_width(width);
        ui.set_min_height(154.0);

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(card_icon(&def)).size(22.0).strong());
                ui.heading(def.name);
            });
            ui.add_space(4.0);
            ui.label(format!("Target: {}", def.target));
            ui.label(def.effect);
            ui.label(format!("Duration: {:.0}s", def.duration));
            ui.label(format!("Stability: {:+.0}", def.city_delta));
        });

        if cooldown > 0.0 {
            ui.label(format!("Cooldown: {:.0}s", cooldown.ceil()));
        } else {
            ui.label("Ready");
        }
        let button_text = if def.kind == CardKind::Active {
            "Run policy"
        } else {
            "Use now"
        };
        if ui
            .add_enabled(
                can_play,
                egui::Button::new(button_text).min_size(egui::vec2(120.0, 26.0)),
            )
            .clicked()
        {
            state.play_hand_card(hand_index);
        }
    });
}

fn card_icon(def: &CardDef) -> &'static str {
    if def.city_delta > 0.0 {
        "♥"
    } else {
        match def.intent {
            CardIntent::Air => "↓",
            CardIntent::City => "♥",
            CardIntent::Health => "+",
        }
    }
}

fn render_mechanism_panel(ui: &mut egui::Ui, state: &SimState, texture_id: egui::TextureId) {
    let width = ui.available_width().max(240.0);
    let image_size = egui::vec2(width, 132.0);
    let response = ui.add(
        egui::Image::new(egui::load::SizedTexture::new(texture_id, image_size))
            .fit_to_exact_size(image_size),
    );
    let rect = response.rect;
    let painter = ui.painter_at(rect);

    let (score, severity, label) = mechanism_deviation(state);
    let direction = if score >= 0.0 {
        egui::vec2(1.0, -0.55)
    } else {
        egui::vec2(-1.0, 0.55)
    }
    .normalized();
    let length = 26.0 + 58.0 * severity;
    let start = rect.center() + egui::vec2(10.0, 16.0) - direction * (length * 0.38);
    let end = start + direction * length;
    let arrow_color = if score >= 0.0 {
        egui::Color32::from_rgb(245, 185, 70)
    } else {
        egui::Color32::from_rgb(95, 190, 235)
    };
    let stroke = egui::Stroke::new(3.0 + 2.5 * severity, arrow_color);
    painter.line_segment([start, end], stroke);

    let side = egui::vec2(-direction.y, direction.x);
    let head_len = 10.0 + 7.0 * severity;
    painter.line_segment(
        [end, end - direction * head_len + side * head_len * 0.55],
        stroke,
    );
    painter.line_segment(
        [end, end - direction * head_len - side * head_len * 0.55],
        stroke,
    );

    let label_rect = egui::Rect::from_min_size(
        rect.left_bottom() + egui::vec2(8.0, -28.0),
        egui::vec2(rect.width() - 16.0, 22.0),
    );
    painter.rect_filled(
        label_rect,
        4.0,
        egui::Color32::from_rgba_unmultiplied(8, 12, 18, 190),
    );
    painter.text(
        label_rect.left_center() + egui::vec2(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        label,
        egui::TextStyle::Small.resolve(ui.style()),
        arrow_color,
    );
}

fn mechanism_deviation(state: &SimState) -> (f32, f32, &'static str) {
    let params = state.effective_params();
    let hour = state.time_of_day as f64;
    let nox = (state.chem.no + state.chem.no2) as f32;
    let o3 = state.chem.o3 as f32;
    let voc = state.chem.voc as f32;
    let sun = (solar_arc(hour) * params.solar_flux) as f32;
    let mix = mixing_coeff(
        hour,
        params.solar_flux,
        params.wind_speed,
        params.inversion_strength,
    ) as f32;

    let o3_bias = (o3 - 55.0) / 70.0;
    let nox_bias = (nox - 120.0) / 240.0;
    let voc_bias = (voc - 150.0) / 260.0;
    let sun_bias = (sun - 0.45) / 0.75;
    let low_mix_bias = (0.00022 - mix) / 0.00022;
    let score = (0.36 * o3_bias
        + 0.24 * nox_bias
        + 0.16 * voc_bias
        + 0.14 * sun_bias
        + 0.10 * low_mix_bias)
        .clamp(-1.0, 1.0);
    let severity = score.abs().clamp(0.0, 1.0);
    let label = if score > 0.18 {
        "High side: ozone-forming pressure"
    } else if score < -0.18 {
        "Low side: clearing / night-storage pressure"
    } else {
        "Near balance"
    };

    (score, severity, label)
}

fn render_policy_choice(ctx: &egui::Context, state: &mut SimState) {
    let Some(choice) = state.game.policy_choice else {
        return;
    };

    let mut selected_replace = None;
    let mut accepted_offer = None;
    egui::Window::new("Policy Choice")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            ui.heading("Replace one policy card");
            ui.label("Choose carefully: the offers solve opposite problems.");
            ui.separator();

            match choice.replace_index {
                None => {
                    ui.label("Select a card in your hand to replace:");
                    for i in 0..state.game.hand.len() {
                        let def = card_def(state.game.hand[i]);
                        if ui.button(def.name).clicked() {
                            selected_replace = Some(i);
                        }
                    }
                }
                Some(index) => {
                    ui.label(format!(
                        "Replacing: {}",
                        card_def(state.game.hand[index]).name
                    ));
                    ui.separator();
                    ui.horizontal(|ui| {
                        for offer_index in 0..2 {
                            let offer = choice.offers[offer_index];
                            let def = card_def(offer);
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.set_min_width(100.0);
                                ui.heading(def.name);
                                ui.label(def.effect);
                                ui.label(format!("Stability: {:+.0}", def.city_delta));
                                if ui.button("Choose").clicked() {
                                    accepted_offer = Some(offer_index);
                                }
                            });
                        }
                    });
                }
            }
        });

    if let Some(index) = selected_replace {
        if let Some(active_choice) = &mut state.game.policy_choice {
            active_choice.replace_index = Some(index);
        }
    }
    if let Some(offer_index) = accepted_offer {
        state.accept_policy_offer(offer_index);
    }
}

fn render_lab_ui(ctx: &egui::Context, state: &mut SimState, surrogate: &mut SurrogateRes) {
    egui::SidePanel::right("controls")
        .min_width(360.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Photochemical Smog Lab");
                        if ui.button("Game mode").clicked() {
                            state.reset_game_round();
                        }
                    });
                    ui.label("Expanded urban ozone chemistry with RK4 and optional ONNX surrogate.");
                    ui.separator();

                    ui.collapsing("Scene presets", |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Morning commute").clicked() {
                                apply_preset(state, 7.4, 0.90, 0.85, 0.18, 23.0, 0.55, 0.18, 0.40, false);
                            }
                            if ui.button("Sunny noon").clicked() {
                                apply_preset(state, 12.8, 0.55, 1.30, 0.22, 33.0, 0.32, 0.25, 0.10, false);
                            }
                            if ui.button("Industrial haze").clicked() {
                                apply_preset(state, 14.0, 0.62, 1.15, 0.18, 31.0, 0.58, 0.90, 0.62, false);
                            }
                            if ui.button("Weekend clearing").clicked() {
                                apply_preset(state, 11.6, 0.38, 1.05, 0.65, 27.0, 0.44, 0.18, 0.08, true);
                            }
                        });
                    });

                    ui.collapsing("Quick interventions", |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Traffic spike").clicked() {
                                state.params.traffic_density = (state.params.traffic_density + 0.12).clamp(0.0, 1.0);
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
                    });

                    ui.separator();
                    ui.heading("Controls");
                    ui.add(egui::Slider::new(&mut state.params.traffic_density, 0.0..=1.0).text("Traffic density"));
                    ui.add(egui::Slider::new(&mut state.params.industrial_emissions, 0.0..=1.0).text("Industrial emissions"));
                    ui.add(egui::Slider::new(&mut state.params.solar_flux, 0.2..=1.5).text("Solar flux / heat"));
                    ui.add(egui::Slider::new(&mut state.params.wind_speed, 0.0..=1.0).text("Wind / ventilation"));
                    ui.add(egui::Slider::new(&mut state.params.temperature_c, 10.0..=40.0).text("Temperature (C)"));
                    ui.add(egui::Slider::new(&mut state.params.humidity, 0.1..=1.0).text("Humidity"));
                    ui.add(egui::Slider::new(&mut state.params.inversion_strength, 0.0..=1.0).text("Inversion / trapping"));
                    ui.checkbox(&mut state.params.weekend_mode, "Weekend traffic pattern");
                    ui.add(egui::Slider::new(&mut state.time_of_day, 0.0..=24.0).text("Time of day (h)"));
                    ui.add(egui::Slider::new(&mut state.clock_speed, 30.0..=700.0).text("Sim speed (s/s)"));
                    ui.checkbox(&mut state.auto_advance, "Auto-advance clock");
                    ui.checkbox(&mut state.paused, "Pause");

                    ui.separator();
                    ui.heading("Solver");
                    ui.label(format!("Chemistry step: {:.0} s", CHEM_DT));
                    let model_present = surrogate.inner.is_some();
                    ui.add_enabled_ui(model_present, |ui| {
                        ui.checkbox(&mut surrogate.enabled, "Neural surrogate (ONNX)");
                    });
                    if !model_present {
                        ui.colored_label(egui::Color32::from_rgb(210, 90, 90), "smog_surrogate.onnx not found - RK4 fallback active");
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
                    ui.label(format!("Regime: {}", chemistry_phase(state)));
                    ui.label(format!("Narrative: {}", scenario_narrative(state)));

                    ui.separator();
                    ui.heading("Drivers");
                    let params = state.effective_params();
                    let hour = state.time_of_day as f64;
                    let traffic_now = traffic_profile(hour, params.traffic_density, params.weekend_mode);
                    let mix_now = mixing_coeff(hour, params.solar_flux, params.wind_speed, params.inversion_strength);
                    let daylight = solar_arc(hour) * params.solar_flux;
                    let temp_factor = temperature_factor(params.temperature_c);
                    let humidity_factor = humidity_factor(params.humidity);
                    let trap_factor = trapping_factor(params.inversion_strength, params.wind_speed);
                    ui.monospace(format!("Daylight factor      {:5.2}", daylight));
                    ui.monospace(format!("Photolysis j1        {:5.4} s^-1", j1(hour, params.solar_flux)));
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
                        ui.label("Traffic emits fresh NOx, industry feeds VOCs, sunlight photolyzes NO2, and warm conditions accelerate ozone production. Wind mixes the box, while inversions trap pollutants near the skyline.");
                    });
                });
        });
}

fn apply_card_effect(card: CardId, params: &mut SmogParams) {
    match card {
        CardId::TrafficLimit => {
            params.traffic_density *= 0.65;
        }
        CardId::FactoryPause => {
            params.industrial_emissions *= 0.45;
        }
        CardId::VocScrubber => {
            params.industrial_emissions *= 0.65;
        }
        CardId::SunlightShield => {
            params.solar_flux = (params.solar_flux * 0.72).clamp(0.2, 1.5);
        }
        CardId::VentilationCorridor => {
            params.wind_speed = (params.wind_speed + 0.40).clamp(0.0, 1.0);
            params.inversion_strength *= 0.72;
        }
        CardId::HealthAdvisory => {}
        CardId::RushHourPriority => {
            params.traffic_density = (params.traffic_density * 1.25 + 0.10).clamp(0.0, 1.0);
        }
        CardId::FactoryOvertime => {
            params.industrial_emissions =
                (params.industrial_emissions * 1.30 + 0.08).clamp(0.0, 1.0);
            params.traffic_density = (params.traffic_density + 0.05).clamp(0.0, 1.0);
        }
        CardId::HeatingSubsidy => {
            params.traffic_density = (params.traffic_density + 0.20).clamp(0.0, 1.0);
            params.industrial_emissions = (params.industrial_emissions + 0.04).clamp(0.0, 1.0);
        }
        CardId::FestivalApproval => {
            params.traffic_density = (params.traffic_density + 0.18).clamp(0.0, 1.0);
            params.industrial_emissions = (params.industrial_emissions + 0.08).clamp(0.0, 1.0);
        }
    }
}

fn apply_incident_effect(id: IncidentId, params: &mut SmogParams) {
    match id {
        IncidentId::HeatWave => {
            params.solar_flux = (params.solar_flux * 1.30).clamp(0.2, 1.5);
            params.temperature_c = (params.temperature_c + 5.0).clamp(10.0, 40.0);
        }
        IncidentId::TrafficJam => {
            params.traffic_density = (params.traffic_density * 1.40 + 0.10).clamp(0.0, 1.0);
        }
        IncidentId::FactoryLeak => {
            params.industrial_emissions =
                (params.industrial_emissions * 1.45 + 0.10).clamp(0.0, 1.0);
        }
        IncidentId::ColdSnap => {
            params.temperature_c = (params.temperature_c - 8.0).clamp(10.0, 40.0);
            params.traffic_density = (params.traffic_density + 0.25).clamp(0.0, 1.0);
            params.industrial_emissions = (params.industrial_emissions + 0.06).clamp(0.0, 1.0);
        }
        IncidentId::TemperatureInversion => {
            params.inversion_strength = (params.inversion_strength * 1.45 + 0.25).clamp(0.0, 1.0);
            params.wind_speed *= 0.55;
        }
        IncidentId::CloudBreak => {
            params.solar_flux = (params.solar_flux * 1.50 + 0.10).clamp(0.2, 1.5);
        }
        IncidentId::FestivalFireworks => {
            params.traffic_density = (params.traffic_density + 0.20).clamp(0.0, 1.0);
            params.industrial_emissions = (params.industrial_emissions + 0.10).clamp(0.0, 1.0);
        }
        IncidentId::WindShift => {
            params.wind_speed *= 0.55;
            params.inversion_strength = (params.inversion_strength + 0.15).clamp(0.0, 1.0);
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn air_safety_index(chem: &ChemState, params: &SmogParams, hour: f64, protection: f32) -> f32 {
    let o3 = chem.o3 as f32;
    let no2 = chem.no2 as f32;
    let nox = (chem.no + chem.no2) as f32;
    let voc = chem.voc as f32;
    let sunlight = (solar_arc(hour) as f32 * params.solar_flux as f32 / 1.5).clamp(0.0, 1.0);

    let o3_risk = smoothstep(45.0, 120.0, o3);
    let no2_risk = smoothstep(25.0, 110.0, no2);
    let voc_risk = smoothstep(120.0, 450.0, voc);
    let chain_risk = voc_risk * smoothstep(30.0, 140.0, nox) * sunlight;

    let mut hazard = 0.50 * o3_risk + 0.25 * no2_risk + 0.15 * chain_risk + 0.10 * voc_risk;
    hazard *= 1.0 - protection.clamp(0.0, 0.35);

    (100.0 - 100.0 * hazard).clamp(0.0, 100.0)
}

fn meter(ui: &mut egui::Ui, label: &str, value: f32, color: egui::Color32) {
    let value = value.clamp(0.0, 100.0);
    ui.vertical(|ui| {
        ui.label(format!("{} {:.0}", label, value));
        ui.add(
            egui::ProgressBar::new(value / 100.0)
                .fill(color)
                .desired_width(165.0),
        );
    });
}

fn grade_label(score: f32) -> &'static str {
    match score as u32 {
        85..=100 => "A",
        70..=84 => "B",
        55..=69 => "C",
        40..=54 => "D",
        _ => "F",
    }
}

fn result_reason(air: f32, city: f32) -> &'static str {
    if air < city - 10.0 {
        "Result: city stayed active, but smog exposure was too high."
    } else if city < air - 10.0 {
        "Result: air improved, but the city was over-controlled."
    } else {
        "Result: balanced response. Clean air and city stability stayed close."
    }
}

fn next_incident_label(state: &SimState) -> String {
    if let Some(next) = state
        .game
        .scheduled_incidents
        .iter()
        .filter(|incident| !incident.triggered)
        .min_by(|a, b| a.start.total_cmp(&b.start))
    {
        format!(
            "Next incident window in {:.0}s",
            (next.start - state.game.round_elapsed).max(0.0)
        )
    } else {
        "No more scheduled incidents.".to_string()
    }
}

fn active_reaction_label(state: &SimState) -> &'static str {
    let params = state.effective_params();
    let sunlight = solar_arc(state.time_of_day as f64) * params.solar_flux;
    let nox = state.chem.no + state.chem.no2;
    if state.chem.no > 35.0 && state.chem.o3 < 35.0 {
        "Titration: NO + O3 -> NO2 + O2"
    } else if state.chem.voc > 150.0 && nox > 55.0 && sunlight > 0.35 {
        "VOC chain: VOC + NOx + sunlight -> O3 buildup"
    } else if sunlight > 0.25 {
        "Photolysis: NO2 + sunlight -> ozone pathway"
    } else {
        "Night storage: weak sunlight, slower ozone formation"
    }
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
    let params = state.effective_params();
    let hour = state.time_of_day as f64;
    let daylight = solar_arc(hour) * params.solar_flux;
    if daylight < 0.05 {
        "Nighttime titration / residual-layer storage"
    } else if params.inversion_strength > 0.65 && params.wind_speed < 0.25 {
        "Trapped urban plume under inversion"
    } else if state.chem.no > 35.0 && state.chem.o3 < 18.0 {
        "Fresh traffic plume suppressing ozone"
    } else if params.wind_speed > 0.75 {
        "Ventilated atmosphere clearing the haze"
    } else if state.chem.o3 > 85.0 {
        "Photochemical smog episode"
    } else {
        "Ozone building under sunlight"
    }
}

fn scenario_narrative(state: &SimState) -> &'static str {
    let params = state.effective_params();
    if params.weekend_mode && params.industrial_emissions > 0.55 {
        "Lower traffic but persistent industrial VOC loading."
    } else if params.temperature_c > 32.0 && params.solar_flux > 1.1 {
        "Hot bright conditions favor fast ozone buildup."
    } else if params.inversion_strength > 0.55 {
        "The boundary layer is trapping pollutants near street level."
    } else if params.wind_speed > 0.6 {
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
