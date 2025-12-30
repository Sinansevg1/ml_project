# ML Project

Bu proje makine öğrenmesi (RandomForest) modeli kullanan bir uygulamadır.

## Kurulum

1. Python 3.7+ yüklü olduğundan emin olun
2. Proje klasöründe aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

## Kullanım

Ana programı çalıştırmak için:

```bash
python main.py
```

Modeli incelemek için:

```bash
python inspect_model.py
```

## Dosyalar

- `main.py` - Ana eğitim ve tahmin kodları
- `inspect_model.py` - Eğitilmiş modeli analiz eden script
- `temiz_veri_v*.csv` - Eğitim verileri
- `rf_model.pkl` - Eğitilmiş RandomForest modeli
