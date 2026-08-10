# Python'un hafif (küçük boyutlu) resmi bir sürümünü temel alıyoruz
FROM python:3.11-slim

# Konteyner içinde çalışma klasörümüz
WORKDIR /app

# Önce sadece requirements.txt'i kopyalayıp kütüphaneleri kuruyoruz.
# Bunu ayrı bir adımda yapmak, kod değiştiğinde Docker'ın kütüphaneleri
# tekrar tekrar indirmemesini sağlar (daha hızlı yeniden build).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Şimdi projenin geri kalanını kopyalıyoruz (app.py, .streamlit/config.toml vb.)
COPY . .

# Streamlit'in varsayılan çalıştığı port
EXPOSE 8501

# Konteyner başladığında çalıştırılacak komut
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
