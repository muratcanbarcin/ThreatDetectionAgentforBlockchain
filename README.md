# Catch Theft Crypto Security

A **Web3 transaction threat-detection** demo built as a Streamlit dashboard. It combines a **rule-based** layer (GoPlus address security), a **Random Forest** model on a **45-dimensional** on-chain feature vector, and optional **Groq LLM** explanations when the risk pipeline fires.

## Features

- **Layer 1 — GoPlus:** Ethereum mainnet address security flags (phishing, malicious behavior, stealing attack), with a demo mode that simulates a blacklisted address.
- **Layer 2 — Random Forest:** Fraud vs. legitimate classification using the same numeric features the model was trained on (median imputation + sklearn pipeline).
- **Layer 3 — Groq LLM:** Short, user-facing security advisory text when blacklist or ML anomaly triggers (requires `GROQ_API_KEY`).
- **UI:** Scenario buttons (safe / known threat / zero-day anomaly), manual feature tuning, verdicts, latency, raw JSON payloads for inspection.

## Repository layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit application (main entry point). |
| `middleware.py` | `ThreatDetectionAgent`: GoPlus fetch, RF inference, Groq prompt. |
| `models/` | Trained artifacts: `rf_model.pkl`, `model_features.pkl` (created by training; not in git). |
| `train_model.py` | Trains RF pipeline; writes files under `models/`. |
| `main.py` | Optional simulation harness and Matplotlib charts (`latency_chart.png`, `resource_chart.png`). |
| `mock_data.py` | Mock addresses for `main.py` scenarios. |
| `data/` | Place `transaction_dataset.csv` here (not committed; see `.gitignore`). |
| `.env.example` | Template for API keys. |

## Requirements

- Python **3.10+** (recommended)
- Network access for GoPlus and optional Groq calls

## Setup

1. **Clone** the repository and create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS/Linux: `source .venv/bin/activate`

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset and model artifacts**

   - Add **`data/transaction_dataset.csv`** (Ethereum fraud-style dataset with a `FLAG` column and numeric features). The training script expects the same layout used when the project was developed.
   - **Train** the model (creates the `models/` directory and writes `rf_model.pkl` and `model_features.pkl` there):

     ```bash
     python train_model.py
     ```

   These `.pkl` files are gitignored; you must run training (or obtain the files) before starting the app.

4. **Environment variables**

   Copy `.env.example` to `.env` and set your [Groq API key](https://console.groq.com/keys):

   ```env
   GROQ_API_KEY=your_key_here
   ```

   If `GROQ_API_KEY` is missing, the app still runs; the agent falls back to a static warning message instead of calling the LLM.

5. **Optional logo**

   Place `logo.png` in the project root to show it in the sidebar; otherwise a text placeholder is used.

## Run the dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

## Optional: simulation and charts

```bash
python main.py
```

Generates bar and pie charts illustrating latency and LLM bypass/trigger rates for the mock scenarios.

## License / disclaimer

This project is intended for **research, education, and demonstration**. On-chain labels and third-party APIs can be incomplete or lag real-world abuse. **Do not rely on this stack as your sole security control** for production fund custody or compliance decisions.
