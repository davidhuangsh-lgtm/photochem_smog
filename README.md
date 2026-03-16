# Photochemical Smog Simulator

An interactive real-time simulation of **photochemical smog formation over a stylized urban environment**, built in **Rust** for the **“Chem is Life”** chemistry show — a collaboration between **AI-Lab** and **ChemiClub**.

This project demonstrates how **real atmospheric chemistry** and **machine learning surrogates** can be combined to create a fast, educational, visually engaging simulation.

---

## What This Project Shows

The simulation models the urban chemistry behind **ground-level ozone** and **photochemical smog**.

In the atmosphere:

- **Nitrogen oxides (NOx)** are emitted by traffic and industry
- **Sunlight** drives photolysis reactions
- **Volatile organic compounds (VOCs)** interact with the NOx cycle
- **Ozone (O3)** can accumulate and form visible smog
- **Wind, humidity, temperature, and atmospheric inversion** affect pollutant buildup and clearing

The result is a real-time visual simulation where the user can change urban and environmental conditions and watch the sky, haze, and pollutant concentrations evolve.

---

## AI + Chemistry Integration

The project has two chemistry solvers:

### 1. Reference ODE Solver
A real chemistry model is integrated numerically using a **fourth-order Runge–Kutta (RK4)** method.

### 2. Neural Network Surrogate
A neural network is trained on many ODE trajectories and exported as an **ONNX** model.

At runtime, the Rust application loads the ONNX model and uses it as a **fast surrogate solver**, replacing repeated expensive chemistry integration steps with neural-network inference.

This reflects a real scientific workflow used in computational atmospheric science:
**train a neural network to emulate a physical solver so the system stays interactive at scale**.

---

## Audience Takeaway

A visitor can adjust:

- traffic level
- solar flux
- wind speed
- industrial emissions
- temperature
- humidity
- inversion strength
- weekday/weekend conditions

and instantly see:

- smog haze increase or clear
- ozone concentration rise
- NO2 coloration change
- the “smog index” worsen or improve

This makes the connection between **everyday urban activity**, **sunlight-driven chemistry**, and **breathable air quality** immediate and visual.

---

# Project Structure

```text
Cargo.toml
train_surrogate.py
src/
  chemistry.rs
  main.rs
  surrogate.rs
```

### File roles

- `train_surrogate.py`  
  Trains the neural surrogate and exports `smog_surrogate.onnx`

- `src/chemistry.rs`  
  Atmospheric chemistry model, emissions, environmental forcing, and RK4 stepping

- `src/surrogate.rs`  
  ONNX Runtime integration and surrogate inference wrapper

- `src/main.rs`  
  Bevy app, GUI, controls, plots, rendering, and simulation loop

- `Cargo.toml`  
  Rust dependencies and project metadata

---

# Requirements

You need:

- **Python 3.10+** recommended
- **pip**
- **Rust**
- **Cargo**
- Internet access for first-time dependency installation

Recommended:
- Windows PowerShell or Command Prompt
- Rust stable toolchain

---

# 1. Install Python

## Check whether Python is already installed

Open **Command Prompt** or **PowerShell** and run:

```bash
python --version
```

If that does not work, try:

```bash
py --version
```

If neither works, install Python.

## Install Python on Windows

1. Go to the official Python website
2. Download Python 3.10 or newer
3. During installation, enable **Add Python to PATH**
4. Complete installation

Then confirm:

```bash
python --version
pip --version
```

If `python` does not work but `py` does, use `py` in the commands below.

---

# 2. Install Rust

Check whether Rust is already installed:

```bash
rustc --version
cargo --version
```

If not installed:

1. Go to the official Rust website
2. Install Rust using `rustup`
3. Restart the terminal after installation

Then verify again:

```bash
rustc --version
cargo --version
```

---

# 3. Download or Clone the Project

If you downloaded a ZIP:
1. Extract it
2. Open a terminal in the extracted project folder

You should be inside the folder containing:

- `Cargo.toml`
- `train_surrogate.py`
- `src/`

Example:

```bash
cd path\to\photochem_smog
```

---

# 4. Create a Python Virtual Environment

A virtual environment keeps Python packages for this project isolated from your system Python.

## On Windows

Create the virtual environment:

```bash
python -m venv .venv
```

If your system uses `py`:

```bash
py -m venv .venv
```

This creates a local folder named `.venv`.

---

# 5. Activate the Virtual Environment

## PowerShell

```bash
.venv\Scripts\Activate.ps1
```

## Command Prompt

```bash
.venv\Scripts\activate.bat
```

After activation, you should usually see `(.venv)` at the beginning of your terminal line.

Example:

```text
(.venv) C:\Users\YourName\photochem_smog>
```

---

# 6. Upgrade pip

Once the virtual environment is active:

```bash
python -m pip install --upgrade pip
```

---

# 7. Install Python Packages

Install the packages used by the training script:

```bash
pip install numpy torch onnx
```

You can verify package installation with:

```bash
pip list
```

---

# 8. Train the Surrogate Model

The Rust app expects an ONNX model file generated by the Python training script.

Run:

```bash
python train_surrogate.py
```

This will train the neural surrogate and export:

```text
smog_surrogate.onnx
```

in the project directory.

## Faster test training

For a quick smoke test, you can reduce the training workload:

```bash
set SMOG_TRAIN_SAMPLES=3000
set SMOG_EPOCHS=3
python train_surrogate.py
```

If you are using PowerShell:

```bash
$env:SMOG_TRAIN_SAMPLES="3000"
$env:SMOG_EPOCHS="3"
python train_surrogate.py
```

---

# 9. Build and Run the Rust Application

After the ONNX model has been created:

```bash
cargo run --release
```

This compiles the simulator in release mode and launches the app.

---

# Full Quickstart

If everything is installed already, the full sequence is:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy torch onnx
python train_surrogate.py
cargo run --release
```

If `python` does not work on your machine, replace it with `py`.

---

# Controls in the App

The right-hand control panel contains simulation controls for:

- traffic density
- solar flux
- wind speed
- industrial emissions
- temperature
- humidity
- inversion strength
- weekend mode
- solver mode / surrogate usage
- presets and interventions

The right-hand panel supports **vertical scrolling**, so if the controls do not all fit on screen, scroll inside the panel.

---

# How the Simulation Works

At each update step, the app evolves atmospheric state variables such as:

- `NO2`
- `NO`
- `O3`
- `VOC`

using environmental and emission drivers like:

- sunlight
- traffic profile
- industrial emissions
- wind-driven ventilation
- inversion trapping
- humidity and temperature effects

The system can be advanced in one of two ways:

## RK4 reference mode
Uses the explicit chemistry equations directly.

## ONNX surrogate mode
Uses the trained neural network to approximate the next chemistry state.

This makes the simulation fast enough for interactive public demonstration.

---

# Common Problems and Fixes

## 1. `python` is not recognized

Try:

```bash
py --version
```

If that works, use `py` instead of `python`.

If neither works, reinstall Python and make sure **Add Python to PATH** is enabled.

---

## 2. `pip` is not recognized

Try:

```bash
python -m pip --version
```

or

```bash
py -m pip --version
```

Then install packages using:

```bash
python -m pip install numpy torch onnx
```

---

## 3. PowerShell blocks virtual environment activation

If PowerShell refuses to activate the venv, run PowerShell as a normal user and use:

```bash
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate again:

```bash
.venv\Scripts\Activate.ps1
```

---

## 4. `smog_surrogate.onnx` not found

You must run:

```bash
python train_surrogate.py
```

before:

```bash
cargo run --release
```

Make sure the ONNX file is created in the project root.

---

## 5. Rust build errors

First check your Rust version:

```bash
rustc --version
cargo --version
```

Then update Rust:

```bash
rustup update
```

After that, retry:

```bash
cargo run --release
```

---

## 6. Bevy query conflict / `B0001`

If you see an ECS query panic mentioning `Sprite` conflicts in `sync_visuals`, use the patched version of `main.rs` where the conflicting queries are wrapped in a `ParamSet`.

---

## 7. Training is too slow

Use smaller temporary training settings:

### Command Prompt
```bash
set SMOG_TRAIN_SAMPLES=3000
set SMOG_EPOCHS=3
python train_surrogate.py
```

### PowerShell
```bash
$env:SMOG_TRAIN_SAMPLES="3000"
$env:SMOG_EPOCHS="3"
python train_surrogate.py
```

Then later run the full training for better surrogate quality.

---

# Recommended Development Workflow

For normal use:

1. Activate virtual environment
2. Retrain surrogate if the chemistry inputs changed
3. Run the Rust app

Example:

```bash
.venv\Scripts\Activate.ps1
python train_surrogate.py
cargo run --release
```

If you modify:
- chemistry variables
- surrogate input size
- surrogate output mapping
- environmental parameters used by the neural model

you must retrain the ONNX model again.

---

# Dependencies

## Rust
- `bevy`
- `bevy_egui`
- `ort`

## Python
- `numpy`
- `torch`
- `onnx`

---

# Example `Cargo.toml`

```toml
[package]
name    = "photochem_smog"
version = "0.1.0"
edition = "2021"

[dependencies]
bevy      = "0.15"
bevy_egui = "0.31"
ort       = { version = "2.0.0-rc.12", features = ["ndarray"] }

[profile.dev]
opt-level = 1
```

---

# Educational Summary

This simulator is designed to communicate three ideas clearly:

1. **Smog is chemistry, not just “dirty air”**
2. **Sunlight, emissions, and weather interact strongly**
3. **AI can accelerate scientific simulation without replacing the physical model behind it**

That combination makes it a strong fit for a public-facing chemistry demonstration.

---

# License / Use
```text
Copyright (c) AI-Lab and ChemiClub
For educational and demonstration use.
```

---

# Credits

Created for the **“Chem is Life”** chemistry show.

A collaboration between:

- **AI-Lab**
- **ChemiClub**

---

# One-Command Reminder

After installation, the project is run as:

```bash
python train_surrogate.py
cargo run --release
```
