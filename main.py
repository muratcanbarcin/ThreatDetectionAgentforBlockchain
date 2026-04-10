import time
import pandas as pd
import matplotlib.pyplot as plt
import random
from mock_data import mock_addresses

def rule_based_filter(address_data):
    """
    Adresin kara listede olup olmadığını kontrol eder.
    Kara listedeyse True (tehdit var) döndürür.
    """
    if address_data.get("in_osint_blacklist") is True:
        return True
    return False

def ml_anomaly_detection(address_data):
    """
    İşlem geçmişindeki anomalileri kontrol eder.
    (Sıfır değerli işlemler, ani hacim artışları vs.)
    Anomali tespiti yapılırsa True döner.
    """
    features = address_data.get("transaction_features", {})
    # Basit bir eşik (threshold) tabanlı ML simülasyonu
    is_spike = features.get("sudden_volume_spike", False)
    high_zero_value = features.get("zero_value_tx_1h", 0) > 10
    
    if is_spike or high_zero_value:
        return True
    return False

def trigger_llm(address_data):
    """
    Yerel LLM analizini simüle etmek için 1.2 saniyelik bir gecikme ekler
    ve güvenlik analizi sonucu döner.
    """
    time.sleep(1.2)
    return "LLM Analizi Tamamlandı: Yüksek riskli ve uyumsuz işlem paterni tespit edildi."

def evaluate_transaction(address_data):
    """
    Ana Karar Mekanizması:
    1. Kural tabanlı filtre ve anomali tespiti devreye girer.
    2. Tehdit bulunursa LLM tetiklenir (Pending/Denied kararı verilir).
    3. Tehdit yoksa LLM pas geçilir (Bypass - Allow).
    """
    is_rule_broken = rule_based_filter(address_data)
    is_anomaly_detected = ml_anomaly_detection(address_data)
    
    llm_triggered = False
    llm_response = ""
    decision = ""
    
    # Tehdit/Anomali algılanması durumunda
    if is_rule_broken or is_anomaly_detected:
        llm_triggered = True
        llm_response = trigger_llm(address_data)
        
        if is_rule_broken:
            decision = "Denied" # Katı kural ihlali
        else:
            decision = "Pending" # Şüpheli ancak tamamen yasaklı değil
    else:
        # Pürüzsüz işlem - LLM Atlanıyor
        decision = "Allow"
        
    return {
        "decision": decision,
        "llm_triggered": llm_triggered,
        "llm_response": llm_response
    }

def visualize_latency(df):
    """
    Pandas DataFrame'deki gecikme (latency) verilerini alır ve
    Matplotlib ile performans grafiği çizer.
    """
    plt.figure(figsize=(9, 6))
    
    # "1200.00 ms" gibi formatlanmış verileri floata çevirelim
    latencies = df["Latency (ms)"].str.replace(" ms", "").astype(float)
    scenarios = df["Scenario Type"]
    
    # Renk paleti tasarımı (Safe -> Yeşil, Malicious -> Kırmızı, Zero-Day -> Turuncu)
    colors = ['#2ca02c', '#d62728', '#ff7f0e']
    
    bars = plt.bar(scenarios, latencies, color=colors, edgecolor='black', alpha=0.8)
    
    # Eksen isimleri ve başlık
    plt.title("Performance Analysis: Processing Latency by Threat Scenario", fontsize=14, pad=15)
    plt.xlabel("Scenario", fontsize=12, labelpad=10)
    plt.ylabel("Processing Time (ms)", fontsize=12, labelpad=10)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(latencies) * 0.02), 
                 f"{yval} ms", ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.ylim(0, max(latencies) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("latency_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[+] İlk grafik başarılı şekilde 'latency_chart.png' adıyla dizine kaydedildi!")

def load_test_and_visualize():
    """
    100 işlemlik bir yük testi çalıştırarak sistemin çoğunlukla LLM'i pas geçip
    (Bypass) ne kadar kaynak tasarrufu (optimizasyon) sağladığını görselleştirir.
    """
    print("\n[!] 100 işlemlik yük testi başlatılıyor...")
    print("    Not: Simülasyon gereği LLM analizleri için yaklaşık 24 saniye beklenecek. Lütfen bekleyin...")
    
    # 1. 100 işlemlik batch üretim (%80 Safe, %10 Malicious, %10 Zero-Day)
    batch = []
    batch.extend([mock_addresses[0]] * 80) # Safe Address
    batch.extend([mock_addresses[1]] * 10) # Known Malicious
    batch.extend([mock_addresses[2]] * 10) # Zero-Day
    
    random.shuffle(batch) # İşlem sırasını rastgele hale getiriyoruz
    
    llm_bypassed = 0
    llm_triggered = 0
    
    # 2. İşlemlerden kararlar toplanıyor (Latency tutmaya gerek yok)
    for tx in batch:
        evaluation = evaluate_transaction(tx)
        if evaluation["llm_triggered"]:
            llm_triggered += 1
        else:
            llm_bypassed += 1
            
    # 3. Pie Chart (Pasta Grafiği) ile görselleştirme
    labels = ['LLM Bypassed\n(Safe / Fast-Track)', 'LLM Triggered\n(Deep Analysis)']
    sizes = [llm_bypassed, llm_triggered]
    
    # Estetik, kurumsal-akademik renk seçimi
    colors = ['#4daf4a', '#e41a1c']  # Muted Green ve Muted Red
    explode = (0.05, 0)  # Safe tarafını %5 dışarı taşıralım (vurgu için)
    
    plt.figure(figsize=(8, 6))
    
    # Dilimlerin içindeki textleri özel formatlayan fonksiyon
    def custom_autopct(pct, allvals):
        absolute = int(round(pct / 100.0 * sum(allvals)))
        return f"{absolute} Transactions\n({pct:.1f}%)"
    
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=labels, 
        colors=colors, 
        autopct=lambda pct: custom_autopct(pct, sizes),
        startangle=140,
        explode=explode,
        shadow=True,
        textprops=dict(color="black", fontweight='bold', fontsize=11)
    )
    
    # Dilimlerin iç yazılarının stilini biraz daha belirgin yapalım
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        
    plt.title("Resource Optimization: LLM Trigger Rate over 100 Transactions", fontsize=15, pad=20, fontweight='bold')
    
    plt.savefig("resource_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[+] Load test tamamlandı. Grafik başarılı şekilde 'resource_chart.png' adıyla kaydedildi!")

def run_tests():
    print("Sistem Başlatılıyor...\nModüller ve kurallar yükleniyor...\n")
    results = []
    scenario_names = ["Safe Address", "Known Malicious", "Zero-Day Suspicious"]
    
    for idx, addr_data in enumerate(mock_addresses):
        start_time = time.time()
        evaluation = evaluate_transaction(addr_data)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000 
        
        results.append({
            "Address": addr_data["address"],
            "Scenario Type": scenario_names[idx] if idx < len(scenario_names) else f"Unknown Scenario {idx}",
            "Final Decision": evaluation["decision"],
            "LLM Triggered?": "Yes" if evaluation["llm_triggered"] else "No",
            "Latency (ms)": f"{latency_ms:.2f} ms"
        })
        
    df = pd.DataFrame(results)
    
    print("--- SİMÜLASYON TEST SONUÇLARI ---")
    print(df.to_string(index=False))
    
    # İlk Chart: Latency Bar Chart
    visualize_latency(df)
    
    # İkinci Chart: Load Test Pie Chart
    load_test_and_visualize()

if __name__ == "__main__":
    run_tests()
