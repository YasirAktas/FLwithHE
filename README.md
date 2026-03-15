# FLwithHE Project

Modüler Federated Learning (FedAvg) örneği + Homomorfik Şifreleme (HE) için altyapı iskeleti.

## Klasör Yapısı
```
src/
  fl/
    partitions.py        # IID ve Dirichlet veri bölme
    client.py            # İstemci eğitimi
    aggregator.py        # Federated averaging + HE kancası
    fedavg_runner.py     # Ana çalışma scripti (modüler)
  models/
    mnist_cnn.py         # MNIST için küçük CNN
    cifar_cnn.py         # CIFAR-10 için ResNet-18 (CIFAR'a uyarlanmış)
  he/
    encryption.py        # PlainContext ve TenSEAL CKKS tabanlı HomomorphicContext
config/
  default.yaml           # Varsayılan hiperparametreler
requirements.txt
README.md
```

## Kurulum
```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

(Opsiyonel) GPU veya özel HE kütüphanesi için resmi kurulum komutlarını ayrıca çalıştırın.

## Adım Adım Çalıştırma (Windows CMD)
1) Proje klasörüne girin
```cmd
cd .\FLwithHE
```

2) (Yoksa) sanal ortam oluşturun
```cmd
python -m venv .venv
```

3) Ortamı aktif edin
```cmd
.\.venv\Scripts\activate
```

4) Bağımlılıkları kurun
```cmd
pip install -r requirements.txt
```

5) Modüler runner 
```cmd
python -m src.fl.fedavg_runner --dataset mnist --num_clients 5 --rounds 5 --local_epochs 1 --partition iid
```

6) CIFAR-10 ile çalıştırma
```cmd
python -m src.fl.fedavg_runner --dataset cifar10 --num_clients 5 --rounds 40 --local_epochs 3 --partition iid --use_aug --weight_decay 0.0005 --scheduler cosine
```

7) Non-IID (Dirichlet) veri dağılımı
```cmd
python -m src.fl.fedavg_runner --dataset mnist --num_clients 5 --rounds 5 --local_epochs 1 --partition dirichlet --dirichlet_alpha 0.3
```
Açıklama:
- `partition=dirichlet`: Sınıf dağılımını istemciler arasında dengesiz (heterojen) yapar.
- `dirichlet_alpha`: Küçük değer → daha heterojen (bazı istemciler bazı sınıfları daha çok görür). Büyük değer → IID'ye yaklaşır.
Örn. `alpha=0.3` daha gerçekçi, heterojen bir dağılım üretir; yakınsama IID'ye göre biraz daha yavaş olabilir.

7) CUDA kapatma/açma
- Kapatma: `--no_cuda`
- Açık bırakmak için ek bir bayrak gerekmez (GPU varsa otomatik kullanılır)

Örnek:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --num_clients 5 --rounds 3 --no_cuda
```

8) Homomorfik Şifreleme (HE) deneyleri

CKKS ile tam model aggregation:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks --payload_mode full_model --compare_reference --save_metrics_csv results/results_ckks_full.csv
```

Paillier ile analytics aggregation:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme paillier --payload_mode analytics --analytics_include_grad_norm --compare_reference --save_metrics_csv results/results_paillier_analytics.csv
```

Notlar:
- `--payload_mode full_model`: istemci model parametreleri şifrelenir ve şifreli FedAvg aggregation yapılır.
- `--payload_mode analytics`: `loss_sum`, `correct_count`, `sample_count` ve opsiyonel `grad_norm` gibi küçük skalerler şifrelenir.
- `--payload_mode integer_stats`: sınıf sayımları gibi tam sayı istatistikleri şifrelenir.
- `--compare_reference`: plaintext aggregation ile decrypted aggregation arasındaki `mean_abs_error` ve `max_abs_error` değerlerini loglar.
- `--save_metrics_csv`: çıktı dosyası yoludur. Göreli yol verirseniz mevcut çalışma klasörüne yazılır. `results/results.csv` gibi verirseniz klasör otomatik oluşturulur.
- Çıktıda `Encrypt`, `Agg`, `Decrypt` ve `Total` süreleri ayrıca raporlanır.

10) Parametre özeti
- `--num_clients`: İstemci sayısı
- `--rounds`: Global tur sayısı
- `--local_epochs`: Her istemcide epoch
- `--batch_size`: Lokal batch boyutu
- `--lr`: Öğrenme oranı
- `--dataset`: `mnist` veya `cifar10`
- `--partition`: `iid` veya `dirichlet`
- `--dirichlet_alpha`: Non-IID şiddeti (küçükse daha heterojen)
- `--use_encryption`: Şifreli toplama modunu tetikler
- `--encryption_scheme`: `ckks` (varsayılan, TenSEAL) veya `paillier` (python-paillier)
- `--payload_mode`: `full_model`, `analytics`, `integer_stats`
- `--analytics_include_grad_norm`: analytics payload'ına `grad_norm` ekler
- `--param_sweep`: Tam model için ilk `N` parametreyi şifreleyerek sweep deneyi yapar. Örnek: `2,5,10,50,100,500`
- `--save_metrics_csv`: metrikleri CSV olarak yazar
- `--compare_reference`: plaintext ve decrypted aggregate farkını loglar
- `--no_cuda`: GPU kullanma

11) Çıktılar
Her turun sonunda:
```
Round XX: Acc=...% Loss=...
```
Global modelin test doğruluğu ve kaybı raporlanır.

12) Tipik hatalar ve çözümler
- MNIST indirme hatası: İnternet bağlantısını kontrol edin, tekrar deneyin.
- CUDA uyarısı: `--no_cuda` kullanarak CPU’da çalıştırın.
- Paket bulunamadı: Ortamın aktif olduğundan emin olun ve `pip install -r requirements.txt` çalıştırın.

13) Temiz çıkış
```cmd
deactivate
```

## Çalıştırma (Modüler Runner)
```cmd
python -m src.fl.fedavg_runner --dataset mnist --num_clients 5 --rounds 5 --local_epochs 1 --partition iid
```
Dirichlet (non-IID) örneği:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --num_clients 5 --rounds 5 --partition dirichlet --dirichlet_alpha 0.3
```
CIFAR-10 örneği:
```cmd
python -m src.fl.fedavg_runner --dataset cifar10 --num_clients 5 --rounds 40 --local_epochs 3 --use_aug --weight_decay 0.0005 --scheduler cosine
```
Encryption (HE) aktif çalıştırma — CKKS / full model:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks --payload_mode full_model --compare_reference --save_metrics_csv results/results_ckks_full.csv
```
Encryption (HE) aktif çalıştırma — Paillier / analytics:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme paillier --payload_mode analytics --analytics_include_grad_norm --compare_reference --save_metrics_csv results/results_paillier_analytics.csv
```
Parameter sweep örneği:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks --payload_mode full_model --param_sweep 2,5,10,50,100,500,1000,5000 --compare_reference --save_metrics_csv results/results_sweep_ckks.csv
```
CUDA kapatmak:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --no_cuda
```




## Homomorfik Şifreleme

- Amaç: FedAvg eğitim akışını değiştirmeden farklı HE payload türlerini deneysel olarak karşılaştırmak.
- Kapsam: Eğitim plaintext kalır; encryption/aggregation deney katmanı `full_model`, `analytics` ve `integer_stats` payload'larını ayrı ayrı ölçer.

### Desteklenen Şemalar

| Şema | Kütüphane | Uygun Payload | Hız | Hassasiyet |
|---|---|---|---|---|
| `ckks` (varsayılan) | TenSEAL | `full_model`, `analytics`, `integer_stats` | Daha yavaş | Yaklaşık (float) |
| `paillier` | python-paillier | `analytics`, `integer_stats`, küçük `full_model` sweep'leri | Daha hızlı | Fixed-point / integer |

### Kurulum

CKKS için:
```cmd
pip install tenseal
```

Paillier için:
```cmd
pip install python-paillier
```

- `requirements.txt` içinde her ikisi de listelidir.

### Nasıl Etkinleştirilir?

CKKS ile tam model:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks --payload_mode full_model --compare_reference --save_metrics_csv results/results_ckks_full.csv
```

Paillier ile analytics:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme paillier --payload_mode analytics --analytics_include_grad_norm --compare_reference --save_metrics_csv results/results_paillier_analytics.csv
```

- Konfigürasyon: [config/default.yaml](config/default.yaml) içinde `use_encryption: true` yapabilirsiniz.

### API Özeti
- [src/he/encryption.py](src/he/encryption.py)
  - `PlainContext`: `encrypt(t)`, `decrypt(t)` no-op; `add(a,b)`, `mul_scalar(a,s)` düz tensör işlemleri.
  - `HomomorphicContext`: TenSEAL CKKS ile çalışır. Parametreler: `poly_modulus_degree` (varsayılan 8192), `coeff_mod_bit_sizes` (60,40,40,60), `global_scale` ($2^{40}$). CKKS slot sayısı `poly_modulus_degree/2`.
  - `PaillierContext`: python-paillier ile çalışır. Additively homomorphic integer/fixed-point payload aggregation için kullanılır.
  - İç temsil: `EncryptedTensor` şifreli parçaların ve orijinal şeklin tutulduğu hafif bir kap.
- [src/fl/aggregator.py](src/fl/aggregator.py)
  - `aggregate_encrypted_dict(...)`: payload dictionary'lerini şifreli toplar ve decrypt sürelerini ayrı döndürür.
- [src/fl/fedavg_runner.py](src/fl/fedavg_runner.py)
  - payload üretimi, reference comparison, parameter sweep ve CSV logging burada yapılır.

### Payload Modları

| `payload_mode` | Açıklama | Tipik kullanım |
|---|---|---|
| `full_model` | Tüm model parametreleri veya sweep ile ilk `N` parametre şifrelenir | CKKS avantajını göstermek |
| `analytics` | `loss_sum`, `correct_count`, `sample_count`, opsiyonel `grad_norm` şifrelenir | Küçük skalerlerde Paillier avantajını göstermek |
| `integer_stats` | `class_counts` gibi tam sayı istatistikleri şifrelenir | Exact integer toplamada Paillier avantajını göstermek |

### Parameter Sweep

Tam model ölçeklenebilirlik deneyleri için:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks --payload_mode full_model --param_sweep 2,5,10,50,100,500,1000,5000 --compare_reference --save_metrics_csv results/results_sweep_ckks.csv
```

- `--param_sweep` yalnızca `payload_mode=full_model` ile çalışır.
- Model update'in ilk `N` flatten edilmiş parametresi şifrelenir.
- CSV'de her round ve her `N` için ayrı satır yazılır.

### Çıktıda Süre Raporlama

Her round sonunda şifreleme, aggregation ve decrypt süreleri ayrı ayrı gösterilir:
```
Round 01: Acc=95.12% Loss=0.1543 | Train=8.21s Encrypt=3.45s Agg=0.92s Decrypt=0.14s | Total=12.58s Elapsed=12.58s
```

Sweep modunda özet satırı yazdırılır; detaylı metrikler CSV'ye gider:
```
Round 01: Acc=95.39% Loss=0.1615 | Train=22.43s Sweep=8 configs | Total=23.66s Elapsed=23.79s
```

### CSV Çıktısı

CSV satırları şu alanları içerir:

`timestamp`, `round`, `dataset`, `model`, `num_clients`, `scheme`, `payload_mode`, `training_time`, `encrypt_time`, `aggregate_time`, `decrypt_time`, `he_total_time`, `total_round_time`, `ciphertext_count`, `encrypted_values`, `payload_nbytes`, `accuracy`, `loss`, `mean_abs_error`, `max_abs_error`

Analytics ve integer payload'larında ayrıca:

`analytics_reference`, `analytics_decrypted`, `integer_reference`, `integer_decrypted`

Not:
- `--save_metrics_csv results_sweep_ckks.csv` derseniz dosya proje köküne yazılır.
- `--save_metrics_csv results/results_sweep_ckks.csv` derseniz dosya `results/` klasörüne yazılır.

### Performans ve Sınırlamalar
- CKKS yaklaşık aritmetik kullanır; küçük sayısal farklar beklenir.
- Paillier tam integer aritmetik kullanır; float parametreler ölçeklenerek dönüştürülür.
- Büyük modellerde slot parçalama nedeniyle ek bellek ve zaman maliyeti oluşur.
- Aynı HE bağlamı (anahtarlar/ölçek) istemci ve sunucu tarafında tutarlı olmalıdır.
- HE açıkken eğitim tur süreleri anlamlı ölçüde artabilir.

### Sorun Giderme
- `ImportError: TenSEAL not installed`: Sanal ortam aktifken `pip install tenseal` çalıştırın.
- `ImportError: phe not installed`: Sanal ortam aktifken `pip install python-paillier` çalıştırın.
- Derleme/kurulum hataları: İlgili kütüphanenin platform kılavuzunu takip edin veya geçici olarak `--use_encryption` olmadan çalıştırın.

---

## PTB-XL ile Çalıştırma

### Adım 1 — Veri Setini İndir

Tarayıcıdan şu adrese git ve ZIP dosyasını indir (~1.8 GB):
```
https://physionet.org/content/ptb-xl/1.0.3/
```

### Adım 2 — ZIP'i Çıkart

İndirilen ZIP'i şu klasöre çıkart:
```
FLwithHE/data/ptbxl/
```

Sonuç şöyle görünmeli:
```
data/
└── ptbxl/
    └── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
        ├── records100/
        ├── records500/
        ├── ptbxl_database.csv
        └── scp_statements.csv
```

### Adım 3 — Gerekli Paketi Kur

```cmd
pip install wfdb
```

### Adım 4 — Dataset'i Test Et

```cmd
python test_ptbxl.py
```

Beklenen çıktı:
```
[PTBXLDataset] train: 19230 kayıt | sınıf dağılımı: {'NORM': 8314, 'MI': 3776, 'STTC': 2994, 'CD': 2971, 'HYP': 1175}
[PTBXLDataset] test:  2158 kayıt  | sınıf dağılımı: {'NORM':  932, 'MI':  411, 'STTC':  351, 'CD':  351, 'HYP':  113}
x shape  : torch.Size([32, 1000, 12])
TEST BAŞARILI
```

### Adım 5 — Modeli Seç ve Çalıştır

**CNN Medium (önerilen, hızlı):**
```cmd
python -m src.fl.fedavg_runner --dataset ptbxl --model cnn_medium --num_clients 5 --rounds 5 --local_epochs 1
```

**CNN Large:**
```cmd
python -m src.fl.fedavg_runner --dataset ptbxl --model cnn_large --num_clients 5 --rounds 5 --local_epochs 1
```

**Logistic Regression (en hızlı, baseline):**
```cmd
python -m src.fl.fedavg_runner --dataset ptbxl --model logistic --num_clients 5 --rounds 5 --local_epochs 1
```
.\.venv\Scripts\python.exe -m src.fl.fedavg_runner --dataset ptbxl --ptbxl_model logistic --num_clients 5 --rounds 5 --local_epochs 1

**Şifreleme ile (Paillier):**
```cmd
python -m src.fl.fedavg_runner --dataset ptbxl --model cnn_medium --num_clients 5 --rounds 5 --use_encryption --encryption_scheme paillier
```

### Sınıf Açıklamaları

| Sınıf | Etiket | Açıklama |
|---|---|---|
| NORM | 0 | Normal ECG |
| MI | 1 | Miyokard Enfarktüsü |
| STTC | 2 | ST/T Değişikliği |
| CD | 3 | İletim Bozukluğu |
| HYP | 4 | Hipertrofi |



