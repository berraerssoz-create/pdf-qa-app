import os
import re
import time
import uuid
import json
import base64
import hashlib
import secrets
import html as html_lib
import io
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# ÇEVİRİLER (Türkçe / İngilizce arayüz metinleri)
# ============================================================
CEVIRILER = {
    "tr": {
        "baslik": "🤖 Kişisel AI Asistanım",
        "gecmis_sohbetler": "Geçmiş Sohbetler",
        "yeni_sohbet": "➕ Yeni Sohbet",
        "yeni_sohbet_adi": "Yeni Sohbet",
        "yeniden_adlandir_giris": "Yeni isim",
        "kaydet": "✔ Kaydet",
        "vazgec": "✕ Vazgeç",
        "tema_baslik": "🎨 Görünüm",
        "tema_pembe": "Pembe",
        "tema_beyaz": "Beyaz",
        "tema_gri": "Gri",
        "tema_koyu": "Koyu",
        "belge_ekle_baslik": "Belge Ekle (opsiyonel)",
        "belge_eklendi_bilgi": "📎 Bir belge eklendi, cevaplar bu belgeye dayanarak verilecek.",
        "belge_kaldir": "✖ Belgeyi kaldır",
        "ozet_baslik": "### Özet",
        "ozetle_buton": "📝 Özetle",
        "ozet_cikariliyor": "Özet çıkarılıyor...",
        "ozet_kisaltildi": "⚠️ İçerik çok uzun olduğu için sadece bir kısmı özetlendi.",
        "icerik_isleniyor": "İçerik işleniyor ve embedding oluşturuluyor...",
        "mesaj_yaz": "Bir şeyler yaz, resim ya da PDF ekle (📎)...",
        "cevap_araniyor": "Cevap aranıyor...",
        "dil_secimi": "🌐",
        "arama_ozelligi_bilgi": "🔎 Bu cevap, güncel bilgi için internetten birden fazla kaynak taranarak zenginleştirildi.",
        "giris_baslik": "🔐 Giriş Yap",
        "giris_sekme": "Giriş Yap",
        "kayit_sekme": "Kayıt Ol",
        "kullanici_adi_etiket": "Kullanıcı adı",
        "sifre_etiket": "Şifre",
        "sifre_tekrar_etiket": "Şifre (tekrar)",
        "giris_yap_buton": "Giriş Yap",
        "kayit_ol_buton": "Kayıt Ol",
        "alanlari_doldur_uyari": "Kullanıcı adı ve şifre girmelisin.",
        "giris_hata": "Kullanıcı adı veya şifre hatalı.",
        "sifre_kisa_uyari": "Şifre en az 4 karakter olmalı.",
        "sifreler_eslesmiyor_uyari": "Şifreler eşleşmiyor.",
        "kayit_basarili": "Kayıt başarılı! Şimdi giriş yapabilirsin.",
        "kullanici_adi_alinmis": "Bu kullanıcı adı zaten alınmış.",
        "cikis_yap_buton": "🚪 Çıkış Yap",
        "belge_ekle_baslik_sidebar": "📎 Belge Ekle",
    },
    "en": {
        "baslik": "🤖 My Personal AI Assistant",
        "gecmis_sohbetler": "Chat History",
        "yeni_sohbet": "➕ New Chat",
        "yeni_sohbet_adi": "New Chat",
        "yeniden_adlandir_giris": "New name",
        "kaydet": "✔ Save",
        "vazgec": "✕ Cancel",
        "tema_baslik": "🎨 Appearance",
        "tema_pembe": "Pink",
        "tema_beyaz": "White",
        "tema_gri": "Gray",
        "tema_koyu": "Dark",
        "belge_ekle_baslik": "Add Document (optional)",
        "belge_eklendi_bilgi": "📎 A document has been added, answers will be based on it.",
        "belge_kaldir": "✖ Remove document",
        "ozet_baslik": "### Summary",
        "ozetle_buton": "📝 Summarize",
        "ozet_cikariliyor": "Generating summary...",
        "ozet_kisaltildi": "⚠️ The content was too long, only part of it was summarized.",
        "icerik_isleniyor": "Processing content and creating embeddings...",
        "mesaj_yaz": "Type a message, attach an image or PDF (📎)...",
        "cevap_araniyor": "Looking for an answer...",
        "dil_secimi": "🌐",
        "arama_ozelligi_bilgi": "🔎 This answer was enriched by searching multiple sources on the web for up-to-date information.",
        "giris_baslik": "🔐 Log In",
        "giris_sekme": "Log In",
        "kayit_sekme": "Sign Up",
        "kullanici_adi_etiket": "Username",
        "sifre_etiket": "Password",
        "sifre_tekrar_etiket": "Password (again)",
        "giris_yap_buton": "Log In",
        "kayit_ol_buton": "Sign Up",
        "alanlari_doldur_uyari": "You must enter a username and password.",
        "giris_hata": "Incorrect username or password.",
        "sifre_kisa_uyari": "Password must be at least 4 characters.",
        "sifreler_eslesmiyor_uyari": "Passwords don't match.",
        "kayit_basarili": "Sign up successful! You can now log in.",
        "kullanici_adi_alinmis": "This username is already taken.",
        "cikis_yap_buton": "🚪 Log Out",
        "belge_ekle_baslik_sidebar": "📎 Add Document",
    },
}


def t(anahtar, **kwargs):
    dil = st.session_state.get("dil", "tr")
    metin = CEVIRILER.get(dil, CEVIRILER["tr"]).get(anahtar, anahtar)
    if kwargs:
        return metin.format(**kwargs)
    return metin


# Bazı PDF'lerde bozuk font/kodlama yüzünden metin çıkarılırken gecersiz
# "surrogate" karakterler olusabiliyor; bunlar utf-8'e cevrilemedigi icin
# Google API'ye gonderilirken hataya neden oluyor. Bu regex onlari temizler.
_SURROGATE_RE = re.compile('[\ud800-\udfff]')


def clean_text(text: str) -> str:
    return _SURROGATE_RE.sub('', text)


class NativeGoogleEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batch_size = 100
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=batch,
                task_type="retrieval_document"
            )
            embeddings.extend(res["embedding"])
            if i + batch_size < len(texts):
                time.sleep(1)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return res["embedding"]


def extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


st.set_page_config(page_title="Kisisel AI Asistanim", page_icon="🤖", layout="wide")

# ============================================================
# TEMA (GÖRÜNÜM) SİSTEMİ
# ============================================================
TEMALAR = {
    "pembe": {
        "bg": "#FDF2F6", "sidebar_bg": "#FCE4EC", "text": "#1F2937",
        "kullanici_balon": "#6366F1", "kullanici_yazi": "#FFFFFF",
        "asistan_balon": "#FFFFFF", "asistan_yazi": "#1F2937",
    },
    "beyaz": {
        "bg": "#FFFFFF", "sidebar_bg": "#F3F4F6", "text": "#1F2937",
        "kullanici_balon": "#6366F1", "kullanici_yazi": "#FFFFFF",
        "asistan_balon": "#F3F4F6", "asistan_yazi": "#1F2937",
    },
    "gri": {
        "bg": "#E9EAEC", "sidebar_bg": "#D8DADD", "text": "#1F2937",
        "kullanici_balon": "#4B5563", "kullanici_yazi": "#FFFFFF",
        "asistan_balon": "#FFFFFF", "asistan_yazi": "#1F2937",
    },
    "koyu": {
        "bg": "#1E1E2E", "sidebar_bg": "#181825", "text": "#E4E4E7",
        "kullanici_balon": "#8B5CF6", "kullanici_yazi": "#FFFFFF",
        "asistan_balon": "#2A2A3C", "asistan_yazi": "#E4E4E7",
    },
}

if "tema" not in st.session_state:
    st.session_state.tema = "pembe"
if "dil" not in st.session_state:
    st.session_state.dil = "tr"

_secilen_tema = TEMALAR[st.session_state.tema]
st.markdown(f"""
<style>
    .stApp {{
        background-color: {_secilen_tema['bg']};
        color: {_secilen_tema['text']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {_secilen_tema['sidebar_bg']};
    }}
    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
    }}
    .stChatInput textarea {{
        border-radius: 12px;
    }}
</style>
""", unsafe_allow_html=True)

col_baslik, col_tema, col_dil = st.columns([5, 1.3, 1])
with col_dil:
    secilen_dil = st.selectbox(
        t("dil_secimi"), ["Türkçe", "English"],
        index=0 if st.session_state.dil == "tr" else 1,
        label_visibility="collapsed", key="dil_secici"
    )
    st.session_state.dil = "tr" if secilen_dil == "Türkçe" else "en"
with col_tema:
    tema_secenekleri = {
        t("tema_pembe"): "pembe", t("tema_beyaz"): "beyaz",
        t("tema_gri"): "gri", t("tema_koyu"): "koyu",
    }
    tema_gosterim = list(tema_secenekleri.keys())
    tema_ters = {v: k for k, v in tema_secenekleri.items()}
    secilen_tema_adi = st.selectbox(
        t("tema_baslik"), tema_gosterim,
        index=tema_gosterim.index(tema_ters[st.session_state.tema]),
        label_visibility="collapsed", key="tema_secici"
    )
    yeni_tema_kodu = tema_secenekleri[secilen_tema_adi]
    if yeni_tema_kodu != st.session_state.tema:
        st.session_state.tema = yeni_tema_kodu
        st.rerun()
with col_baslik:
    st.title(t("baslik"))

if "GOOGLE_API_KEY" not in st.secrets:
    st.error(
        "API anahtarı bulunamadı. Yerelde çalıştırıyorsan proje klasöründe "
        ".streamlit/secrets.toml dosyası oluşturup içine "
        'GOOGLE_API_KEY = "senin-anahtarin" yazmalısın. Streamlit Cloud\'da '
        "ise Ayarlar -> Secrets kısmından ekle."
    )
    st.stop()

from supabase import create_client

google_api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = google_api_key
genai.configure(api_key=google_api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=google_api_key
)

# ============================================================
# SOHBETLER (Supabase veritabanı - kalıcı ve bulut tabanlı)
# ============================================================
if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error(
        "Supabase bağlantı bilgileri bulunamadı. .streamlit/secrets.toml "
        'dosyasına SUPABASE_URL = "..." ve SUPABASE_KEY = "..." eklemelisin.'
    )
    st.stop()


@st.cache_resource
def supabase_baglantisi():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def sohbetleri_yukle():
    """Veritabanındaki tüm sohbetleri getirir. {chat_id: {"title":..., "history":[...]}}"""
    sb = supabase_baglantisi()
    sonuc = sb.table("sohbetler").select("*").execute()
    return {satir["chat_id"]: satir["data"] for satir in sonuc.data}


def aktif_sohbeti_kaydet():
    """Sadece şu an aktif olan sohbeti veritabanına yazar (tamamını değil,
    böylece her mesajda tüm sohbetleri yeniden göndermemize gerek kalmıyor)."""
    sb = supabase_baglantisi()
    sb.table("sohbetler").upsert({
        "chat_id": st.session_state.current_chat_id,
        "data": aktif_sohbet,
    }).execute()


def sohbeti_sil(chat_id):
    sb = supabase_baglantisi()
    sb.table("sohbetler").delete().eq("chat_id", chat_id).execute()


def google_arama_ile_yanitla(soru_metni, gecmis_liste):
    """Genel sohbet modunda, Google Arama ile gerçek zamanlı ve birden
    fazla kaynaktan bilgiye dayanarak cevap üretmeyi dener. Bu özellik
    (API/kütüphane sürümüne bağlı olarak) başarısız olursa None döner;
    çağıran taraf bu durumda normal (aramasız) cevaba düşer."""
    try:
        model = genai.GenerativeModel(
            "gemini-flash-latest",
            tools=[{"google_search": {}}]
        )
        gemini_gecmisi = []
        for soru, cevap in gecmis_liste:
            gemini_gecmisi.append({"role": "user", "parts": [soru]})
            gemini_gecmisi.append({"role": "model", "parts": [cevap]})
        sohbet = model.start_chat(history=gemini_gecmisi)
        sonuc = sohbet.send_message(soru_metni)
        return sonuc.text
    except Exception:
        return None


# ============================================================
# BELGE (PDF) İŞLEME
# ============================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_source_id" not in st.session_state:
    st.session_state.processed_source_id = None
if "full_text" not in st.session_state:
    st.session_state.full_text = None
if "summary" not in st.session_state:
    st.session_state.summary = None

MAX_SUMMARY_CHARS = 300000


def pdf_isle(pdf_dosyasi):
    kaynak_id = "pdf:" + pdf_dosyasi.name
    if st.session_state.processed_source_id == kaynak_id:
        return
    with st.spinner(t("icerik_isleniyor")):
        with open("gecici.pdf", "wb") as f:
            f.write(pdf_dosyasi.getvalue())

        loader = PyPDFLoader("gecici.pdf")
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        full_text = "\n\n".join(doc.page_content for doc in docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_texts = text_splitter.split_text(full_text)

        embeddings = NativeGoogleEmbeddings(api_key=google_api_key)
        st.session_state.vectorstore = Chroma.from_texts(texts=split_texts, embedding=embeddings)
        st.session_state.processed_source_id = kaynak_id
        st.session_state.full_text = full_text
        st.session_state.summary = None


# ============================================================
# KULLANICI GİRİŞİ / KAYDI (Supabase "kullanicilar" tablosu)
# ============================================================
def sifre_hashle(sifre: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    hash_str = hashlib.sha256((salt + sifre).encode("utf-8")).hexdigest()
    return hash_str, salt


def sifre_dogrula(sifre: str, salt: str, hash_str: str) -> bool:
    return hashlib.sha256((salt + sifre).encode("utf-8")).hexdigest() == hash_str


def kullanici_kayit_ol(kullanici_adi: str, sifre: str, email: str):
    sb = supabase_baglantisi()
    mevcut = sb.table("kullanicilar").select("kullanici_adi").eq("kullanici_adi", kullanici_adi).execute()
    if mevcut.data:
        return False, t("kullanici_adi_alinmis")
    hash_str, salt = sifre_hashle(sifre)
    sb.table("kullanicilar").insert({
        "kullanici_adi": kullanici_adi,
        "sifre_hash": hash_str,
        "salt": salt,
        "email": email,
        "otp_aktif": True,
    }).execute()
    return True, t("kayit_basarili")


def kullanici_giris_yap(kullanici_adi: str, sifre: str) -> bool:
    sb = supabase_baglantisi()
    sonuc = sb.table("kullanicilar").select("*").eq("kullanici_adi", kullanici_adi).execute()
    if not sonuc.data:
        return False
    kayit = sonuc.data[0]
    return sifre_dogrula(sifre, kayit["salt"], kayit["sifre_hash"])


def kullanici_admin_mi(kullanici_adi: str) -> bool:
    sb = supabase_baglantisi()
    sonuc = sb.table("kullanicilar").select("admin_mi").eq("kullanici_adi", kullanici_adi).execute()
    if sonuc.data and sonuc.data[0].get("admin_mi"):
        return True
    return False


# ============================================================
# OTP / 2FA (e-posta ile dogrulama kodu)
# ============================================================
def kullanici_otp_bilgisi(kullanici_adi: str):
    sb = supabase_baglantisi()
    sonuc = (
        sb.table("kullanicilar")
        .select("email, otp_secret, otp_aktif, otp_kod_son_tarih, otp_son_dogrulama")
        .eq("kullanici_adi", kullanici_adi)
        .execute()
    )
    if sonuc.data:
        return sonuc.data[0]
    return {
        "email": None, "otp_secret": None, "otp_aktif": False,
        "otp_kod_son_tarih": None, "otp_son_dogrulama": None,
    }


def otp_guvenilir_mi(kullanici_adi: str, gun_sayisi: int = 10) -> bool:
    """Kullanici son 'gun_sayisi' gun icinde basariyla OTP dogruladiysa,
    tekrar kod sormaya gerek olmadigini soyler."""
    bilgi = kullanici_otp_bilgisi(kullanici_adi)
    son_dogrulama_str = bilgi.get("otp_son_dogrulama")
    if not son_dogrulama_str:
        return False
    try:
        son_dogrulama = datetime.fromisoformat(son_dogrulama_str.replace("Z", "+00:00"))
    except ValueError:
        return False
    if son_dogrulama.tzinfo is None:
        son_dogrulama = son_dogrulama.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - son_dogrulama < timedelta(days=gun_sayisi)


def otp_dogrulamayi_kaydet(kullanici_adi: str):
    sb = supabase_baglantisi()
    sb.table("kullanicilar").update({
        "otp_son_dogrulama": datetime.now(timezone.utc).isoformat(),
    }).eq("kullanici_adi", kullanici_adi).execute()


def email_kod_gonder(alici_email: str, kod: str):
    gonderen = st.secrets["EMAIL_SENDER"]
    uygulama_sifresi = st.secrets["EMAIL_APP_PASSWORD"]

    mesaj = MIMEText(f"Giris dogrulama kodunuz: {kod}\n\nBu kod 5 dakika gecerlidir.")
    mesaj["Subject"] = "Kisisel AI Asistanim - Giris Dogrulama Kodu"
    mesaj["From"] = gonderen
    mesaj["To"] = alici_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gonderen, uygulama_sifresi)
        server.sendmail(gonderen, [alici_email], mesaj.as_string())


def otp_kod_uret_ve_gonder(kullanici_adi: str, email: str):
    kod = f"{secrets.randbelow(1000000):06d}"
    son_tarih = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    sb = supabase_baglantisi()
    sb.table("kullanicilar").update({
        "otp_secret": kod,
        "otp_kod_son_tarih": son_tarih,
    }).eq("kullanici_adi", kullanici_adi).execute()
    email_kod_gonder(email, kod)


def otp_kodu_dogrula(kullanici_adi: str, girilen_kod: str) -> bool:
    bilgi = kullanici_otp_bilgisi(kullanici_adi)
    kayitli_kod = bilgi.get("otp_secret")
    son_tarih_str = bilgi.get("otp_kod_son_tarih")
    if not kayitli_kod or not son_tarih_str:
        return False
    try:
        son_tarih = datetime.fromisoformat(son_tarih_str.replace("Z", "+00:00"))
    except ValueError:
        return False
    if son_tarih.tzinfo is None:
        son_tarih = son_tarih.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > son_tarih:
        return False
    dogru_mu = girilen_kod.strip() == kayitli_kod
    if dogru_mu:
        otp_dogrulamayi_kaydet(kullanici_adi)
    return dogru_mu


def otp_aktiflestir(kullanici_adi: str, email: str):
    sb = supabase_baglantisi()
    sb.table("kullanicilar").update({
        "email": email,
        "otp_aktif": True,
    }).eq("kullanici_adi", kullanici_adi).execute()


def otp_kapat(kullanici_adi: str):
    sb = supabase_baglantisi()
    sb.table("kullanicilar").update({
        "otp_aktif": False,
    }).eq("kullanici_adi", kullanici_adi).execute()


if "giris_yapildi" not in st.session_state:
    st.session_state.giris_yapildi = False
if "aktif_kullanici" not in st.session_state:
    st.session_state.aktif_kullanici = None
if "otp_bekleniyor" not in st.session_state:
    st.session_state.otp_bekleniyor = False
if "otp_bekleyen_kullanici" not in st.session_state:
    st.session_state.otp_bekleyen_kullanici = None

if st.session_state.otp_bekleniyor:
    st.markdown("""
    <style>
        .stApp { background-color: #E9EAEC; }
    </style>
    """, unsafe_allow_html=True)

    col_bos1, col_orta, col_bos2 = st.columns([1, 1.2, 1])
    with col_orta:
        st.markdown(
            """
            <div style="background-color:#D8DADD; padding:32px; border-radius:16px; margin-top:60px;">
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 🔐 İki Adımlı Doğrulama")
        st.caption("E-postana gönderilen 6 haneli kodu gir (5 dakika geçerli).")
        otp_kod_girisi = st.text_input("Doğrulama kodu", key="otp_kod_girisi", max_chars=6)
        col_dogrula, col_iptal = st.columns(2)
        with col_dogrula:
            if st.button("Doğrula", key="otp_dogrula_buton"):
                if otp_kodu_dogrula(st.session_state.otp_bekleyen_kullanici, otp_kod_girisi):
                    st.session_state.giris_yapildi = True
                    st.session_state.aktif_kullanici = st.session_state.otp_bekleyen_kullanici
                    st.session_state.otp_bekleniyor = False
                    st.session_state.otp_bekleyen_kullanici = None
                    st.rerun()
                else:
                    st.error("Kod yanlış ya da süresi dolmuş, tekrar dene.")
        with col_iptal:
            if st.button("Vazgeç", key="otp_iptal_buton"):
                st.session_state.otp_bekleniyor = False
                st.session_state.otp_bekleyen_kullanici = None
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if not st.session_state.giris_yapildi:
    st.markdown("""
    <style>
        .stApp { background-color: #E9EAEC; }
    </style>
    """, unsafe_allow_html=True)

    col_bos1, col_orta, col_bos2 = st.columns([1, 1.2, 1])
    with col_orta:
        st.markdown(
            """
            <div style="background-color:#D8DADD; padding:32px; border-radius:16px; margin-top:60px;">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"#### {t('giris_baslik')}")

        sekme_giris, sekme_kayit = st.tabs([t("giris_sekme"), t("kayit_sekme")])

        with sekme_giris:
            giris_kadi = st.text_input(t("kullanici_adi_etiket"), key="giris_kadi")
            giris_sifre = st.text_input(t("sifre_etiket"), type="password", key="giris_sifre")
            if st.button(t("giris_yap_buton"), key="giris_buton"):
                if not giris_kadi or not giris_sifre:
                    st.warning(t("alanlari_doldur_uyari"))
                elif kullanici_giris_yap(giris_kadi, giris_sifre):
                    otp_bilgi = kullanici_otp_bilgisi(giris_kadi)
                    if otp_bilgi.get("otp_aktif") and otp_bilgi.get("email") and not otp_guvenilir_mi(giris_kadi):
                        with st.spinner("Kod e-postana gönderiliyor..."):
                            otp_kod_uret_ve_gonder(giris_kadi, otp_bilgi["email"])
                        st.session_state.otp_bekleniyor = True
                        st.session_state.otp_bekleyen_kullanici = giris_kadi
                        st.rerun()
                    else:
                        st.session_state.giris_yapildi = True
                        st.session_state.aktif_kullanici = giris_kadi
                        st.rerun()
                else:
                    st.error(t("giris_hata"))

        with sekme_kayit:
            kayit_kadi = st.text_input(t("kullanici_adi_etiket"), key="kayit_kadi")
            kayit_email = st.text_input("E-posta", key="kayit_email")
            kayit_sifre = st.text_input(t("sifre_etiket"), type="password", key="kayit_sifre")
            kayit_sifre2 = st.text_input(t("sifre_tekrar_etiket"), type="password", key="kayit_sifre2")
            if st.button(t("kayit_ol_buton"), key="kayit_buton"):
                if not kayit_kadi or not kayit_sifre or not kayit_email:
                    st.warning(t("alanlari_doldur_uyari"))
                elif "@" not in kayit_email:
                    st.warning("Geçerli bir e-posta adresi gir.")
                elif len(kayit_sifre) < 4:
                    st.warning(t("sifre_kisa_uyari"))
                elif kayit_sifre != kayit_sifre2:
                    st.warning(t("sifreler_eslesmiyor_uyari"))
                else:
                    basarili, mesaj = kullanici_kayit_ol(kayit_kadi, kayit_sifre, kayit_email)
                    if basarili:
                        st.success(mesaj)
                    else:
                        st.error(mesaj)

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


with st.sidebar:
    st.caption(f"👤 {st.session_state.aktif_kullanici}")
    if st.button(t("cikis_yap_buton")):
        st.session_state.giris_yapildi = False
        st.session_state.aktif_kullanici = None
        st.rerun()

    with st.expander("📧 İki Adımlı Doğrulama (2FA)"):
        otp_bilgi_aktif = kullanici_otp_bilgisi(st.session_state.aktif_kullanici)

        if otp_bilgi_aktif.get("otp_aktif"):
            st.success(f"2FA aktif ✅ ({otp_bilgi_aktif.get('email')})")
            if st.button("2FA'yı Kapat", key="otp_kapat_buton"):
                otp_kapat(st.session_state.aktif_kullanici)
                st.rerun()
        else:
            st.caption("Etkinleştirirsen, her girişte e-postana bir doğrulama kodu gönderilir.")
            otp_email_girisi = st.text_input(
                "E-posta adresin", value=otp_bilgi_aktif.get("email") or "", key="otp_email_girisi"
            )
            if st.button("2FA'yı Etkinleştir", key="otp_baslat_buton"):
                if otp_email_girisi and "@" in otp_email_girisi:
                    otp_aktiflestir(st.session_state.aktif_kullanici, otp_email_girisi)
                    st.success("2FA etkinleştirildi!")
                    st.rerun()
                else:
                    st.warning("Geçerli bir e-posta adresi gir.")

    if kullanici_admin_mi(st.session_state.aktif_kullanici):
        with st.expander("🛠️ Admin Paneli"):
            sb_admin = supabase_baglantisi()

            kullanicilar_listesi = (
                sb_admin.table("kullanicilar")
                .select("kullanici_adi, created_at, admin_mi")
                .order("created_at")
                .execute()
                .data
            )
            sohbet_sonuc = sb_admin.table("sohbetler").select("chat_id", count="exact").execute()
            toplam_sohbet = sohbet_sonuc.count if sohbet_sonuc.count is not None else len(sohbet_sonuc.data)

            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.metric("Kullanıcı", len(kullanicilar_listesi))
            with col_k2:
                st.metric("Sohbet", toplam_sohbet)

            st.markdown("**Kullanıcılar**")
            for k in kullanicilar_listesi:
                tarih = (k.get("created_at") or "")[:10]
                rozet = " 👑" if k.get("admin_mi") else ""
                st.caption(f"{k['kullanici_adi']}{rozet} — {tarih}")

    st.divider()

    if "chats" not in st.session_state:
        st.session_state.chats = sohbetleri_yukle()
        st.session_state.current_chat_id = next(iter(st.session_state.chats), None)

    def yeni_sohbet_olustur():
        yeni_id = str(uuid.uuid4())
        st.session_state.chats[yeni_id] = {"title": t("yeni_sohbet_adi"), "history": []}
        st.session_state.current_chat_id = yeni_id
        sb = supabase_baglantisi()
        sb.table("sohbetler").upsert({
            "chat_id": yeni_id,
            "data": st.session_state.chats[yeni_id],
        }).execute()

    if not st.session_state.chats or st.session_state.current_chat_id not in st.session_state.chats:
        yeni_sohbet_olustur()

    st.header(t("gecmis_sohbetler"))
    if st.button(t("yeni_sohbet")):
        yeni_sohbet_olustur()

    for chat_id, chat in st.session_state.chats.items():
        duzenleme_anahtari = f"duzenle_{chat_id}"

        if st.session_state.get(duzenleme_anahtari, False):
            yeni_ad = st.text_input(
                t("yeniden_adlandir_giris"), value=chat["title"],
                key=f"yeniden_ad_input_{chat_id}", label_visibility="collapsed"
            )
            col_kaydet, col_vazgec = st.columns(2)
            with col_kaydet:
                if st.button(t("kaydet"), key=f"kaydet_{chat_id}"):
                    chat["title"] = yeni_ad.strip() or chat["title"]
                    st.session_state[duzenleme_anahtari] = False
                    sb = supabase_baglantisi()
                    sb.table("sohbetler").upsert({"chat_id": chat_id, "data": chat}).execute()
                    st.rerun()
            with col_vazgec:
                if st.button(t("vazgec"), key=f"vazgec_{chat_id}"):
                    st.session_state[duzenleme_anahtari] = False
                    st.rerun()
        else:
            secili_mi = chat_id == st.session_state.current_chat_id
            etiket = ("👉 " if secili_mi else "") + chat["title"]

            col_sohbet, col_duzenle, col_sil = st.columns([4, 1, 1])
            with col_sohbet:
                if st.button(etiket, key=f"chat_btn_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
            with col_duzenle:
                if st.button("✏️", key=f"chat_duzenle_{chat_id}"):
                    st.session_state[duzenleme_anahtari] = True
                    st.rerun()
            with col_sil:
                if st.button("🗑️", key=f"chat_sil_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    sohbeti_sil(chat_id)
                    if st.session_state.current_chat_id == chat_id:
                        if st.session_state.chats:
                            st.session_state.current_chat_id = next(iter(st.session_state.chats))
                        else:
                            yeni_sohbet_olustur()
                    st.rerun()

    st.divider()
    st.markdown(f"#### {t('belge_ekle_baslik_sidebar')}")

    sidebar_pdf = st.file_uploader(
        t("belge_ekle_baslik"), type=["pdf"], key="sidebar_pdf_uploader",
        label_visibility="collapsed",
    )
    if sidebar_pdf is not None:
        pdf_isle(sidebar_pdf)

    belge_var_mi = st.session_state.vectorstore is not None

    if belge_var_mi:
        st.info(t("belge_eklendi_bilgi"))
        if st.button(t("belge_kaldir")):
            st.session_state.vectorstore = None
            st.session_state.processed_source_id = None
            st.session_state.full_text = None
            st.session_state.summary = None
            st.rerun()

    if st.button(t("ozetle_buton"), disabled=not belge_var_mi, key="ozetle_buton_sidebar"):
        with st.spinner(t("ozet_cikariliyor")):
            text_for_summary = st.session_state.full_text
            truncated = len(text_for_summary) > MAX_SUMMARY_CHARS
            if truncated:
                text_for_summary = text_for_summary[:MAX_SUMMARY_CHARS]

            summary_prompt_template = ChatPromptTemplate.from_template(
                "Asagidaki icerigi Turkce olarak, maddeler halinde, net ve oz sekilde ozetle. "
                "Ana konuyu, onemli noktalari ve varsa sonuclari vurgula.\n\n"
                "Icerik:\n{context}"
            )
            summary_prompt = summary_prompt_template.format(context=text_for_summary)

            summary_response = llm.invoke(summary_prompt)
            st.session_state.summary = extract_text(summary_response)
            st.session_state.summary_truncated = truncated

    if st.session_state.summary:
        st.markdown(t("ozet_baslik"))
        st.write(st.session_state.summary)
        if st.session_state.get("summary_truncated"):
            st.caption(t("ozet_kisaltildi"))

aktif_sohbet = st.session_state.chats[st.session_state.current_chat_id]


def gecmis_ogesini_oku(item):
    if isinstance(item, dict):
        return item.get("soru", ""), item.get("resimler", []), item.get("cevap", "")
    soru, cevap = item
    return soru, [], cevap


def mesaj_balonu_ciz(rol: str, metin: str, resimler_b64=None, alt_not: str = None):
    """WhatsApp/Claude tarzı sohbet balonu çiziyor: kullanıcı mesajları sağdan,
    asistan cevapları soldan."""
    tema = TEMALAR[st.session_state.tema]
    kullanici_mi = rol == "user"
    hizalama = "flex-end" if kullanici_mi else "flex-start"
    balon_rengi = tema["kullanici_balon"] if kullanici_mi else tema["asistan_balon"]
    yazi_rengi = tema["kullanici_yazi"] if kullanici_mi else tema["asistan_yazi"]

    resim_html = ""
    for resim_b64 in (resimler_b64 or []):
        resim_html += (
            f'<img src="data:image/png;base64,{resim_b64}" '
            f'style="max-width:220px;border-radius:10px;display:block;margin-bottom:6px;" />'
        )

    metin_html = html_lib.escape(metin).replace("\n", "<br>") if metin else ""

    st.markdown(f"""
    <div style="display:flex; justify-content:{hizalama}; margin:8px 0;">
        <div style="max-width:70%; background-color:{balon_rengi}; color:{yazi_rengi};
                    padding:10px 16px; border-radius:16px; word-wrap:break-word;
                    box-shadow:0 1px 2px rgba(0,0,0,0.08);">
            {resim_html}{metin_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if alt_not:
        st.markdown(f"""
        <div style="display:flex; justify-content:{hizalama}; margin:-4px 0 4px 0;">
            <div style="max-width:70%; font-size:0.8em; opacity:0.65; padding:0 16px;">
                {html_lib.escape(alt_not)}
            </div>
        </div>
        """, unsafe_allow_html=True)


belge_modu = st.session_state.vectorstore is not None

st.markdown(f"### {aktif_sohbet['title']}")

for item in aktif_sohbet["history"]:
    soru, resimler, cevap = gecmis_ogesini_oku(item)
    if soru or resimler:
        mesaj_balonu_ciz("user", soru, resimler)
    mesaj_balonu_ciz("assistant", cevap)

# Claude/ChatGPT tarzı mesaj kutusu: ataç (📎) ile hem resim hem PDF
# ekleyebiliyorsun, aynı kutuya yazı da yazabiliyorsun.
mesaj_girisi = st.chat_input(
    t("mesaj_yaz"),
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp", "pdf"]
)

if mesaj_girisi:
    user_question = mesaj_girisi.text or ""
    tum_dosyalar = mesaj_girisi.files or []
    yuklenen_gorseller = [f for f in tum_dosyalar if not f.name.lower().endswith(".pdf")]
    yuklenen_pdfler = [f for f in tum_dosyalar if f.name.lower().endswith(".pdf")]

    if yuklenen_pdfler:
        pdf_isle(yuklenen_pdfler[0])
        belge_modu = st.session_state.vectorstore is not None

    gorsel_b64_listesi = [base64.b64encode(f.getvalue()).decode("utf-8") for f in yuklenen_gorseller]

    kullanici_gosterim_metni = user_question
    for f in yuklenen_pdfler:
        kullanici_gosterim_metni = f"📄 {f.name}\n{kullanici_gosterim_metni}".strip()

    mesaj_balonu_ciz("user", kullanici_gosterim_metni, gorsel_b64_listesi)

    with st.spinner(t("cevap_araniyor")):
        arama_kullanildi = False
        if gorsel_b64_listesi:
            icerik_parcalari = []
            if user_question:
                icerik_parcalari.append({"type": "text", "text": user_question})
            for gorsel_b64 in gorsel_b64_listesi:
                icerik_parcalari.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{gorsel_b64}"
                })
            response = llm.invoke([HumanMessage(content=icerik_parcalari)])
            answer = extract_text(response)
        elif belge_modu and not user_question:
            answer = "Belgeyi yukledim, simdi ne ogrenmek istedigini yazabilir misin?"
        elif belge_modu:
            relevant_docs = st.session_state.vectorstore.similarity_search(user_question, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt_template = ChatPromptTemplate.from_template(
                "Asagidaki icerige dayanarak soruyu yanitla. Cevap icerikte yoksa acikca belirt.\n\n"
                "Icerik:\n{context}\n\nSoru: {question}"
            )
            prompt = prompt_template.format(context=context, question=user_question)
            response = llm.invoke(prompt)
            answer = extract_text(response)
        else:
            gecmis_liste = []
            for item in aktif_sohbet["history"]:
                soru, _, cevap = gecmis_ogesini_oku(item)
                gecmis_liste.append((soru, cevap))

            arama_cevabi = google_arama_ile_yanitla(user_question, gecmis_liste)
            if arama_cevabi is not None:
                answer = arama_cevabi
                arama_kullanildi = True
            else:
                conversation = []
                for soru, cevap in gecmis_liste:
                    conversation.append(HumanMessage(content=soru))
                    conversation.append(AIMessage(content=cevap))
                conversation.append(HumanMessage(content=user_question))
                response = llm.invoke(conversation)
                answer = extract_text(response)

    mesaj_balonu_ciz(
        "assistant", answer,
        alt_not=t("arama_ozelligi_bilgi") if arama_kullanildi else None
    )

    aktif_sohbet["history"].append({
        "soru": user_question,
        "resimler": gorsel_b64_listesi,
        "cevap": answer,
    })

    baslik_ilk_kez_degisti = aktif_sohbet["title"] == t("yeni_sohbet_adi")
    if baslik_ilk_kez_degisti and user_question:
        yeni_baslik = user_question.strip()
        if len(yeni_baslik) > 40:
            yeni_baslik = yeni_baslik[:40] + "..."
        aktif_sohbet["title"] = yeni_baslik

    aktif_sohbeti_kaydet()

    if baslik_ilk_kez_degisti and user_question:
        st.rerun()
