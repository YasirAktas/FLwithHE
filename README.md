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
  he/
    encryption.py        # PlainContext ve gelecekte HE context
config/
  default.yaml           # Varsayılan hiperparametreler
FedAvg_Mnist.py          # İlk tek dosya örneği (korundu)
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
cd C:\Users\yasir\vscodeide\FLwithHE
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

5) Hızlı test (orijinal tek dosya)
```cmd
python FedAvg_Mnist.py --num_clients 2 --rounds 1 --local_epochs 1 --no_cuda
```

6) Modüler runner (önerilen)
```cmd
python -m src.fl.fedavg_runner --num_clients 5 --rounds 5 --local_epochs 1 --partition iid
```

7) Non-IID (Dirichlet) veri dağılımı
```cmd
python -m src.fl.fedavg_runner --partition dirichlet --dirichlet_alpha 0.3
```

8) CUDA kapatma/açma
- Kapatma: `--no_cuda`
- Açık bırakmak için ek bir bayrak gerekmez (GPU varsa otomatik kullanılır)

Örnek:
```cmd
python -m src.fl.fedavg_runner --num_clients 5 --rounds 3 --no_cuda
```

9) Şifreleme kancası (şimdilik stub)
```cmd
python -m src.fl.fedavg_runner --use_encryption
```
Gerçek HE entegrasyonu sonrası `src/he/encryption.py` içindeki `HomomorphicContext` metodları doldurulacaktır.

10) Parametre özeti
- `--num_clients`: İstemci sayısı
- `--rounds`: Global tur sayısı
- `--local_epochs`: Her istemcide epoch
- `--batch_size`: Lokal batch boyutu
- `--lr`: Öğrenme oranı
- `--partition`: `iid` veya `dirichlet`
- `--dirichlet_alpha`: Non-IID şiddeti (küçükse daha heterojen)
- `--use_encryption`: (stub) şifreli toplama modunu tetikler
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
python -m src.fl.fedavg_runner --num_clients 5 --rounds 5 --local_epochs 1 --partition iid
```
Dirichlet (non-IID) örneği:
```cmd
python -m src.fl.fedavg_runner --num_clients 5 --rounds 5 --partition dirichlet --dirichlet_alpha 0.3
```
Encryption (şimdilik stub, gerçek HE eklenince):
```cmd
python -m src.fl.fedavg_runner --use_encryption
```
CUDA kapatmak:
```cmd
python -m src.fl.fedavg_runner --no_cuda
```

## Config Dosyası Kullanımı
`config/default.yaml` içindekileri manuel parametre geçerek override edebilirsiniz. İsterseniz ileri aşamada bir `load_config` fonksiyonu ekleyip YAML dosyasını otomatik okuyabilirsiniz.

## Geliştirme Yol Haritası
1. Gerçek HE entegrasyonu (TenSEAL veya Pyfhel).
2. Şifreli ağırlık toplama (encrypt -> add -> decrypt).
3. İstemci tarafında gizlilik metrikleri / farklılaştırılmış gizlilik (DP) ekleme.
4. Eğitim istatistiklerinin kaydı (CSV / TensorBoard).
5. Testler: Küçük yapay veri ile hızlı birim test.

## Lisans
Eğer ders projesi ise ders yönergelerine uygun bir lisans/metin ekleyin.

## Notlar
- `FedAvg_Mnist.py` basit referans olarak tutuldu.
- Homomorfik şifreleme henüz "no-op" şeklinde; gerçek kütüphane geldiğinde `HomomorphicContext` metodlarını doldurun.
