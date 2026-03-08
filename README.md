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

8) Homomorfik Şifreleme (HE) ile çalıştırma

CKKS (varsayılan, TenSEAL tabanlı — tam model şifreleme):
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks
```

Paillier (python-paillier tabanlı — yalnızca son katman şifrelenir, daha hızlı):
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme paillier
```

Notlar:
- `--use_encryption` aktifken istemci güncellemeleri HE ile şifrelenir ve sunucu tarafında ağırlıklı ortalama şifreli olarak hesaplanır; yalnızca birleştirilmiş sonuç çözülür.
- `ckks`: Tüm model parametrelerini şifreler. TenSEAL kurulu olmalıdır.
- `paillier`: Yalnızca son katman (classifier) parametrelerini şifreler; diğer katmanlar düz metin kalır. `python-paillier` kurulu olmalıdır: `pip install python-paillier`.
- Şifreleme süresi çıktıda `Encrypt=...s` olarak, aggregation süresi `Agg=...s` olarak raporlanır.

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
Encryption (HE) aktif çalıştırma — CKKS:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks
```
Encryption (HE) aktif çalıştırma — Paillier:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme paillier
```
CUDA kapatmak:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --no_cuda
```




## Homomorfik Şifreleme

- Amaç: İstemci ağırlık güncellemelerini gizlilik için şifreleyerek sunucunun bunları göremeden federated averaging yapması.
- Kapsam: İstemci `state_dict` tensörleri şifrelenir; toplayıcı tarafında ağırlıklandırma (`mul_scalar`) ve toplama (`add`) şifreli yapılır; sonuç yalnızca birleşik ağırlıklı ortalama sonrası çözülür.

### Desteklenen Şemalar

| Şema | Kütüphane | Şifrelenen Parametreler | Hız | Hassasiyet |
|---|---|---|---|---|
| `ckks` (varsayılan) | TenSEAL | Tüm model | Yavaş | Yaklaşık (float) |
| `paillier` | python-paillier | Yalnızca son katman | Daha hızlı | Tam (integer) |

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

CKKS ile:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme ckks
```

Paillier ile (son katman şifrelemesi, daha hızlı):
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption --encryption_scheme paillier
```

- Konfigürasyon: [config/default.yaml](config/default.yaml) içinde `use_encryption: true` yapabilirsiniz.

### API Özeti
- [src/he/encryption.py](src/he/encryption.py)
  - `PlainContext`: `encrypt(t)`, `decrypt(t)` no-op; `add(a,b)`, `mul_scalar(a,s)` düz tensör işlemleri.
  - `HomomorphicContext`: TenSEAL CKKS ile çalışır. Parametreler: `poly_modulus_degree` (varsayılan 8192), `coeff_mod_bit_sizes` (60,40,40,60), `global_scale` ($2^{40}$). CKKS slot sayısı `poly_modulus_degree/2`.
  - `PaillierContext`: python-paillier ile çalışır. Yalnızca son katman (`classifier`, `linear`, `model.fc`) parametrelerini şifreler; diğer katmanlar düz metin olarak iletilir.
  - İç temsil: `EncryptedTensor` şifreli parçaların ve orijinal şeklin tutulduğu hafif bir kap.

### Çıktıda Süre Raporlama

Her round sonunda şifreleme ve aggregation süreleri ayrı ayrı gösterilir:
```
Round 01: Acc=95.12% Loss=0.1543 | Train=8.21s Encrypt=3.45s Agg=0.92s | Total=12.58s Elapsed=12.58s
```

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




