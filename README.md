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
    cifar_cnn.py         # CIFAR-10 için CNN
  he/
    encryption.py        # PlainContext ve gelecekte HE context
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
python -m src.fl.fedavg_runner --dataset cifar10 --num_clients 5 --rounds 5 --local_epochs 1 --partition iid
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

8) Şifreleme kancası (şimdilik stub)
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption
```
Gerçek HE entegrasyonu sonrası `src/he/encryption.py` içindeki `HomomorphicContext` metodları doldurulacaktır.

10) Parametre özeti
- `--num_clients`: İstemci sayısı
- `--rounds`: Global tur sayısı
- `--local_epochs`: Her istemcide epoch
- `--batch_size`: Lokal batch boyutu
- `--lr`: Öğrenme oranı
- `--dataset`: `mnist` veya `cifar10`
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
python -m src.fl.fedavg_runner --dataset mnist --num_clients 5 --rounds 5 --local_epochs 1 --partition iid
```
Dirichlet (non-IID) örneği:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --num_clients 5 --rounds 5 --partition dirichlet --dirichlet_alpha 0.3
```
CIFAR-10 örneği:
```cmd
python -m src.fl.fedavg_runner --dataset cifar10 --num_clients 5 --rounds 5 --local_epochs 1
```
Encryption (şimdilik stub, gerçek HE eklenince):
```cmd
python -m src.fl.fedavg_runner --dataset mnist --use_encryption
```
CUDA kapatmak:
```cmd
python -m src.fl.fedavg_runner --dataset mnist --no_cuda
```




