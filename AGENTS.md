# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

This repo contains a photochemical smog simulator for an educational chemistry demo.

- `src/` is the Rust Bevy application.
- `src/chemistry.rs` contains the reference atmospheric chemistry model and RK4 stepping.
- `src/surrogate.rs` loads `smog_surrogate.onnx` through ONNX Runtime and predicts one 10-second chemistry step.
- `src/main.rs` owns the Bevy app, egui controls, game mode, visual sync, and simulation loop.
- `train_surrogate.py` trains the neural surrogate and exports `smog_surrogate.onnx`.
- `target/` and local dependency folders are generated output.

Keep the chemistry constants, Python training model, Rust surrogate input layout, and ONNX model contract aligned. The surrogate input is currently 13 values and the output is 4 concentration values.

## Primary Commands

Rust app:

```powershell
cargo fmt --check
cargo check
cargo run --release
```

Use `cargo run --release` when you need to launch the interactive Bevy simulator. It opens a desktop window and can take time on first build.

Python surrogate:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy torch onnx
python train_surrogate.py
```

Fast smoke training:

```powershell
$env:SMOG_TRAIN_SAMPLES="3000"
$env:SMOG_EPOCHS="3"
python train_surrogate.py
```

## Development Rules

- Prefer small, targeted changes. This is a demo app, so preserve clarity and live-demo reliability over clever abstractions.
- Do not edit `target/`, `node_modules/`, or generated build artifacts.
- Do not manually patch `smog_surrogate.onnx`. Regenerate it with `train_surrogate.py`.
- If you change chemistry equations, state variables, parameter ranges, `CHEM_DT`, or the surrogate input/output layout, update both Rust and Python paths and retrain the ONNX model.
- Keep units explicit in chemistry code. Concentrations are ppb, chemistry integration time is seconds, and daily drivers use decimal hours.
- Avoid adding new runtime dependencies unless they clearly reduce complexity.
- Keep public-facing labels concise and understandable for a chemistry-show audience.

## Rust Conventions

- Use Rust 2021 idioms and the existing Bevy 0.15 / bevy_egui 0.31 style.
- Run `cargo fmt` before finalizing Rust edits.
- Prefer deterministic simulation behavior where possible; if randomness is needed, keep it seeded or controlled as in `GameState`.
- Be careful with Bevy ECS queries. If a system needs multiple mutable queries over related components, use `ParamSet` to avoid runtime query conflicts.
- Keep `ChemState` and `SmogParams` as the central chemistry data contract.
- Clamp physical state and UI-controlled parameters at boundaries, following the existing code pattern.

## Python Surrogate Rules

- `train_surrogate.py` intentionally duplicates parts of `src/chemistry.rs`. When changing the reference model, mirror the change deliberately instead of only editing one side.
- Preserve the ONNX names expected by Rust: input name `input`, output name `output`.
- Preserve the current model export file name unless the Rust loader is updated too: `smog_surrogate.onnx`.
- Use environment variables for training-size experiments rather than hardcoding temporary values.

## Verification Checklist

For Rust-only changes:

```powershell
cargo fmt --check
cargo check
```

For chemistry or surrogate contract changes:

```powershell
cargo fmt --check
cargo check
$env:SMOG_TRAIN_SAMPLES="3000"
$env:SMOG_EPOCHS="3"
python train_surrogate.py
cargo run --release
```

Document any command you could not run and why.

## Known Project Notes

- The Rust app falls back to RK4 if `smog_surrogate.onnx` is absent.
- The README may contain mojibake from previous encoding issues. Avoid spreading those corrupted characters into new docs.
- This project is built for a public-facing educational demo, so UX text should explain visible cause and effect rather than expose implementation jargon.
