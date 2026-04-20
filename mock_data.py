# mock_data.py
# Mock on-chain-style address records for the simulation (treated as if fetched from external APIs).

mock_addresses = [
    # -------------------------------------------------------------
    # Scenario 1: "Safe Address"
    # - Not on OSINT blacklists.
    # - Normal, steady transaction history.
    # - No zero-value spam bursts or sudden volume spikes.
    # - Older / more established account or contract age.
    # -------------------------------------------------------------
    {
        "address": "0x1A2b3C4D5e6F7890aBcDeF1234567890aBCdEf12",
        "in_osint_blacklist": False,
        "osint_tags": [],
        "contract_age_days": 1250,
        "transaction_features": {
            "tx_count_24h": 12,
            "tx_volume_usd_24h": 450.75,
            "sudden_volume_spike": False,
            "zero_value_tx_1h": 0,
            "pattern": "regular",
        },
    },
    # -------------------------------------------------------------
    # Scenario 2: "Known Malicious"
    # - Present on OSINT blacklists (e.g. Chainabuse).
    # - Previously reported as phishing or scam.
    # -------------------------------------------------------------
    {
        "address": "0xBaD0BaD0BaD0BaD0BaD0BaD0BaD0BaD0BaD0BaD0",
        "in_osint_blacklist": True,
        "osint_tags": ["phishing", "scam", "reported"],
        "contract_age_days": 45,
        "transaction_features": {
            "tx_count_24h": 850,
            "tx_volume_usd_24h": 150000.00,
            "sudden_volume_spike": True,
            "zero_value_tx_1h": 15,
            "pattern": "drainer",
        },
    },
    # -------------------------------------------------------------
    # Scenario 3: "Zero-Day Suspicious"
    # - Not on OSINT yet (novel threat).
    # - Behavioral anomalies: many zero-value transfers or extreme volume spikes.
    # - Very new account / contract age.
    # -------------------------------------------------------------
    {
        "address": "0xZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0",
        "in_osint_blacklist": False,
        "osint_tags": [],
        "contract_age_days": 2,
        "transaction_features": {
            "tx_count_24h": 5000,
            "tx_volume_usd_24h": 850000.00,
            "sudden_volume_spike": True,
            "zero_value_tx_1h": 1200,
            "pattern": "burst_anomalous",
        },
    },
]
