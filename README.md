# Hisse Senedi Fiyat Tahmini ve Haber Analizi Projesi

## Proje Hakkında

Bu proje, hisse senedi fiyatlarını tahmin etmek ve haberlerin fiyat üzerindeki etkisini analiz etmek için geliştirilmiş bir web uygulamasıdır. Proje, makine öğrenmesi algoritmaları (Random Forest ve LSTM) kullanarak geçmiş fiyat verilerini analiz eder ve gelecekteki fiyat hareketlerini tahmin eder. Ayrıca, haber analizi yaparak bu haberlerin hisse senedi fiyatları üzerindeki etkisini değerlendirir.

## Özellikler

- **Hisse Senedi Fiyat Tahmini**: Geçmiş verilere dayanarak gelecekteki fiyat hareketlerini tahmin eder
- **Haber Analizi**: Haberlerin hisse senedi fiyatları üzerindeki etkisini analiz eder
- **Gerçek Zamanlı Veri**: Yahoo Finance API'si üzerinden güncel hisse senedi verilerini çeker
- **İnteraktif Grafikler**: Fiyat hareketlerini ve tahminleri görsel olarak sunar
- **Kullanıcı Dostu Arayüz**: Kolay kullanılabilir web arayüzü

## Kullanılan Teknolojiler

### Backend
- **Python**: Ana programlama dili
- **Flask**: Web uygulaması framework'ü
- **TensorFlow**: Derin öğrenme için
- **Scikit-learn**: Makine öğrenmesi algoritmaları için
- **Pandas**: Veri işleme için
- **NumPy**: Matematiksel işlemler için
- **Yahoo Finance API**: Hisse senedi verilerini çekmek için
- **Filterpy**: Kalman filtresi uygulamak için

### Frontend
- **HTML/CSS**: Web arayüzü için
- **JavaScript**: İnteraktif özellikler için
- **Chart.js**: Grafikleri çizmek için
- **Bootstrap**: Responsive tasarım için
- **jQuery**: AJAX istekleri için

## Kurulum

### Gereksinimler
- Python 3.7 veya üzeri
- pip (Python paket yöneticisi)
- Git (opsiyonel)

### Adımlar

1. Projeyi klonlayın veya indirin:
   ```
   git clone https://github.com/kullaniciadi/hisse-senedi-tahmin.git
   cd hisse-senedi-tahmin
   ```

2. Sanal ortam oluşturun ve etkinleştirin:
   ```
   python -m venv venv
   # Windows için:
   venv\Scripts\activate
   # Linux/Mac için:
   source venv/bin/activate
   ```

3. Gerekli paketleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

4. Uygulamayı çalıştırın:
   ```
   python app.py
   ```

5. Tarayıcınızda `http://localhost:5000` adresine gidin.

## Kullanım

1. Ana sayfada hisse senedi sembolünü girin (örneğin: AAPL, GOOGL, MSFT)
2. "Analiz Et" butonuna tıklayın
3. Sistem verileri analiz edecek ve sonuçları gösterecektir:
   - Fiyat grafiği
   - Güncel fiyat ve tahmin edilen fiyat
   - Fiyat değişimi yüzdesi
   - Haber etkisi

## Proje Yapısı

```
hisse-senedi-tahmin/
├── app.py                  # Ana Flask uygulaması
├── tahmin.py               # Tahmin algoritmaları
├── TSLA_DUYGU.py           # Haber analizi modülü
├── news_analyzer.py        # Haber analizi sınıfı
├── requirements.txt        # Gerekli Python paketleri
├── README.md               # Proje dokümantasyonu
├── templates/              # HTML şablonları
│   └── index.html          # Ana sayfa şablonu
└── static/                 # Statik dosyalar (CSS, JS, resimler)
```

## Algoritma Açıklaması

### Veri Hazırlama
- Yahoo Finance API'sinden hisse senedi verileri çekilir
- Kalman filtresi uygulanarak gürültü azaltılır
- Teknik göstergeler (hareketli ortalamalar) hesaplanır
- Haber analizi yapılarak duygu puanları hesaplanır

### Tahmin Modelleri
1. **Random Forest Regressor**:
   - Ağaç sayısı: 200
   - Özellikler: Açılış, Yüksek, Düşük, Hacim, MA5, MA20, Haber Duygu Puanı, Son Haber Var mı

2. **LSTM (Long Short-Term Memory)**:
   - Katmanlar: 2 LSTM katmanı (64 ve 32 nöron), 2 Dense katmanı (16 ve 1 nöron)
   - Dropout: 0.2 (aşırı öğrenmeyi önlemek için)
   - Sequence uzunluğu: 30 gün
   - Batch size: 256
   - Epochs: 50

3. **Ensemble Tahmin**:
   - Random Forest tahminine %80 ağırlık
   - LSTM tahminine %20 ağırlık

### Haber Analizi
- Haberler TextBlob kütüphanesi ile analiz edilir
- Duygu puanı -1 ile 1 arasında hesaplanır (-1: çok negatif, 0: nötr, 1: çok pozitif)
- Bu puan, tahmin modellerine girdi olarak verilir

## Performans İyileştirmeleri

- CPU optimizasyonları için TensorFlow thread ayarları yapılandırıldı
- Batch size artırılarak eğitim hızı iyileştirildi
- Model mimarisi basitleştirilerek hesaplama yükü azaltıldı
- Kalman filtresi ile gürültü azaltıldı

## Gelecek Geliştirmeler

- Daha fazla teknik gösterge eklenebilir
- Daha gelişmiş haber analizi algoritmaları kullanılabilir
- Kullanıcı hesapları ve kişiselleştirilmiş tahminler eklenebilir
- Mobil uygulama geliştirilebilir
- Daha fazla hisse senedi verisi ve daha uzun geçmiş veriler kullanılabilir

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## İletişim

Sorularınız veya önerileriniz için: [email@example.com](mailto:email@example.com)

---

**Not**: Bu proje eğitim amaçlıdır ve finansal tavsiye niteliği taşımaz. Yatırım kararları için profesyonel danışmanlık almanız önerilir.
