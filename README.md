# Catch Theft

**Catch Theft** is a transaction threat-detection demo delivered as a **Streamlit** dashboard. It layers a **rule-based** check (GoPlus address security on Ethereum mainnet), a **Random Forest** trained on a **45-dimensional** on-chain feature vector (**`predict_proba`** + configurable risk threshold), and optional **Groq LLM** explanations when the risk path runs.

## Features

- **Layer 1 — GoPlus:** Address security flags (phishing, malicious behavior, stealing attack). The **Known Phishing** sidebar profile simulates a blacklisted address (GoPlus skipped in demo).
- **Layer 2 — Random Forest:** Fraud vs. legitimate scoring with probability output; UI **strictness slider** maps to `risk_threshold` passed through `ThreatDetectionAgent.check_anomaly` / `evaluate_transaction`.
- **Layer 3 — Groq LLM:** Short security advisory when blacklist or ML anomaly triggers (requires `GROQ_API_KEY`).
- **UI — three tabs:**
  - **Live Threat Analysis:** Methodology expander (summary metrics), scan workflow, verdicts, XAI radar (when applicable), synthetic **transaction trace graph** (NetworkX + Plotly), PDF export, raw vector / payload expanders.
  - **AI Model Analytics:** Global feature importances, illustrative **ROC curve**, held-out **confusion matrix** heatmap.
  - **Integration & Audit Logs:** In-memory **session audit trail** (CSV export) and **developer API** cookbook (`curl` / Python).
- **Test profiles:** Sidebar **select box** loads full 45-feature vectors + wallet context (normal low/active volume, phishing demo, zero-day anomaly, dormant abuse).
- **Tooling:** `utils.py` holds Plotly/PDF/dataset helpers; `mock_data.py` holds profile presets for the UI.

## Repository layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit application (main entry point): layout, session state, tabs, sidebar. |
| `middleware.py` | `ThreatDetectionAgent`: GoPlus, RF inference (`predict_proba`), anomaly threshold, Groq prompts. |
| `utils.py` | Shared helpers: CSV profile sampling, radar/network/PDF builders, synthetic ROC figure. |
| `mock_data.py` | `PROFILE_OPTIONS` and `resolve_test_profile()` for sidebar test presets. |
| `models/` | Trained artifacts: `rf_model.pkl`, `model_features.pkl` (produced by training; typically gitignored). |
| `train_model.py` | Trains the sklearn pipeline; writes files under `models/`. |
| `main.py` | Optional Matplotlib simulation harness (latency / LLM statistics charts). |
| `data/` | Place `transaction_dataset.csv` here (training + UI profile sampling; often gitignored). |
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

   Includes **Streamlit**, **Plotly**, **NetworkX**, **fpdf2**, **scikit-learn**, **groq**, **requests**, **python-dotenv**, and related stack.

3. **Dataset and model artifacts**

   - Add **`data/transaction_dataset.csv`** (fraud-style dataset with a `FLAG` column and the numeric feature columns expected by training). The shipped training path matches the layout used when the project was developed.
   - **Train** the model (creates `models/` and writes `rf_model.pkl` and `model_features.pkl`):

     ```bash
     python train_model.py
     ```

   These `.pkl` files are usually gitignored; run training (or copy in artifacts) before `streamlit run`.

4. **Environment variables**

   Copy `.env.example` to `.env` and set your [Groq API key](https://console.groq.com/keys):

   ```env
   GROQ_API_KEY=your_key_here
   ```

   If `GROQ_API_KEY` is missing, the app still runs; the agent uses a static fallback message instead of calling Groq.

5. **Optional logo**

   Place **`logo.png`** in the project root for the sidebar; otherwise a **Catch Theft** text header is shown.

## Run the dashboard

```bash
streamlit run app.py
```

Open the local URL from the terminal (typically `http://localhost:8501`).

## Optional: simulation and charts

```bash
python main.py
```

Generates plots (e.g. latency and LLM bypass/trigger style summaries) for offline experimentation. This entry point is separate from the Streamlit UI.

## License / disclaimer

Catch Theft is intended for **research, education, and demonstration**. On-chain labels and third-party APIs can be incomplete or lag real-world abuse. **Do not rely on this stack as your sole security control** for production fund custody or compliance decisions.
