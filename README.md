# NeuraCare 2.0 — Edge AI Maternal Health Monitor

A mobile-first maternal healthcare monitoring system built for rural India.
Runs entirely offline on the patient's phone. No cloud. No internet required.

## What it does

- Connects to an ESP32 wearable via Bluetooth and reads heart rate,
  SpO₂, temperature, and stress index every 2 seconds
- A hybrid ML model (GradientBoosting + rules) predicts risk level —
  LOW / MODERATE / HIGH — and auto-escalates critical cases to a doctor
- BioNutriSense engine gives personalised meal recommendations using
  WHO–ICMR guidelines, trimester stage, and locally available regional foods
- Three-role system: Mother, Healthcare Worker, Doctor with a closed-loop
  alert workflow (Pending → Acknowledged → Escalated → Closed)
- Falls back to realistic mock data when no device is connected

## ML Models

| Model | Purpose | Accuracy |

| Vitals Risk (GBT distilled) | LOW / MEDIUM / HIGH risk from vitals | 99.8% |
| Iron urgency | Fine / Needed / Urgent | 99.9% |
| Protein urgency | Fine / Needed / Urgent | 99.9% |
| Hydration urgency | Fine / Needed / Urgent | 99.9% |
| Folic acid urgency | Fine / Needed / Urgent | 100% |

All models run on-device in under 0.1ms. No API calls.

## Tech Stack

- Single file HTML/CSS/JS — no framework, no build step
- Web Bluetooth API for ESP32 BLE communication
- Scikit-learn (Python) for model training
- Distilled to logistic regression weights for JS inference
- localStorage for offline auto-login and patient data

## Hardware

- ESP32 microcontroller
- MAX30102 — heart rate and SpO₂ sensor
- MLX90614 — infrared temperature sensor

## Files

| File | Description |

| `neuracare app.html` | Complete mobile app — open in Chrome to run |
| `neuracare_ml.py` | Python training pipeline for vitals risk model |
| `neuracare_weights.json` | Trained vitals model weights (JS inference) |
| `nutrition_ml_weights.json` | Trained nutrition model weights (JS inference) |

## How to run

Just open `index.html` in Chrome on Android or Chrome desktop.
No installation. No server. No dependencies.

For ESP32 connection — requires Chrome on Android with Bluetooth enabled.

## Regions supported

Tamil Nadu · Kerala · Karnataka · Andhra Pradesh · Maharashtra ·
Gujarat · Rajasthan · Punjab · West Bengal · Odisha

Each region has a curated list of locally available iron, protein,
hydration, calcium, and folic acid-rich foods.

## Disclaimer

NeuraCare 2.0 is an early warning and preventive monitoring system.
It is not a diagnostic tool. All alerts must be verified by a
qualified healthcare professional.


