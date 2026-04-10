# mock_data.py
# Simülasyonda kullanılacak sahte (mock) blokzincir adres verileri
# Bu veriler dış API'lardan (Etherscan, Chainabuse vb.) çekilmiş gibi ele alınacaktır.

mock_addresses = [
    # -------------------------------------------------------------
    # Senaryo 1: "Safe Address" (Güvenli Adres)
    # - OSINT kara listelerinde (blacklist) YOK.
    # - İşlem geçmişi normal ve düzenli.
    # - Sıfır değerli (zero-value) işlem veya aniden artan bir işlem hacmi YOK.
    # - Sözleşme/Hesap yaşı (contract age) eski/güvenilir.
    # -------------------------------------------------------------
    {
        "address": "0x1A2b3C4D5e6F7890aBcDeF1234567890aBCdEf12",
        "in_osint_blacklist": False,
        "osint_tags": [],
        "contract_age_days": 1250,  # 3 yıldan eski
        "transaction_features": {
            "tx_count_24h": 12,
            "tx_volume_usd_24h": 450.75,
            "sudden_volume_spike": False,
            "zero_value_tx_1h": 0,
            "pattern": "regular"
        }
    },

    # -------------------------------------------------------------
    # Senaryo 2: "Known Malicious" (Bilinen Kötü Niyetli)
    # - OSINT (ör. Chainabuse) kara listelerinde kesinlikle VAR.
    # - Daha önce "phishing" veya "scam" olarak raporlanmış.
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
            "pattern": "drainer"
        }
    },

    # -------------------------------------------------------------
    # Senaryo 3: "Zero-Day Suspicious" (Sıfırıncı Gün Şüphelisi)
    # - OSINT kara listelerinde YOK (yeni bir tehdit olduğu için henüz bilinmiyor).
    # - Ancak işlem özelliklerinde anomali VAR: Son 1 saat içinde çok sayıda sıfır
    #   değerli (zero-value) transfer veya normalin çok ötesinde ani hacim artışı.
    # - Hesap çok yeni (contract age çok düşük).
    # -------------------------------------------------------------
    {
        "address": "0xZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0ZeR0",
        "in_osint_blacklist": False,
        "osint_tags": [],  # İstihbarat temiz ama davranış şüpheli
        "contract_age_days": 2,  # Yeni açılmış/deploy edilmiş
        "transaction_features": {
            "tx_count_24h": 5000,
            "tx_volume_usd_24h": 850000.00,
            "sudden_volume_spike": True,
            "zero_value_tx_1h": 1200,  # Tipik bir airdrop scam veya spam phishing saldırısı modeli
            "pattern": "burst_anomalous"
        }
    }
]
