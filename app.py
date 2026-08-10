import os
import re
import time
import uuid
import json
import base64
import html as html_lib
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

google_api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = google_api_key
genai.configure(api_key=google_api_key)

# ============================================================
# SOHBETLER (kullanıcı sistemi olmadan, tek bir ortak kayıt dosyası)
# ============================================================
SOHBET_DOSYASI = "sohbetler.json"


def sohbetleri_kaydet():
    veri = {
        "chats": st.session_state.chats,
        "current_chat_id": st.session_state.current_chat_id,
    }
    with open(SOHBET_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False)


def sohbetleri_yukle():
    if not os.path.exists(SOHBET_DOSYASI):
        return {}, None
    try:
        with open(SOHBET_DOSYASI, "r", encoding="utf-8") as f:
            veri = json.load(f)
        return veri.get("chats", {}), veri.get("current_chat_id")
    except Exception:
        return {}, None


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


with st.sidebar:
    if "chats" not in st.session_state:
        yuklenen_chats, yuklenen_id = sohbetleri_yukle()
        st.session_state.chats = yuklenen_chats
        st.session_state.current_chat_id = yuklenen_id

    def yeni_sohbet_olustur():
        yeni_id = str(uuid.uuid4())
        st.session_state.chats[yeni_id] = {"title": t("yeni_sohbet_adi"), "history": []}
        st.session_state.current_chat_id = yeni_id
        sohbetleri_kaydet()

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
                    sohbetleri_kaydet()
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
                    sohbetleri_kaydet()
            with col_duzenle:
                if st.button("✏️", key=f"chat_duzenle_{chat_id}"):
                    st.session_state[duzenleme_anahtari] = True
                    st.rerun()
            with col_sil:
                if st.button("🗑️", key=f"chat_sil_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        if st.session_state.chats:
                            st.session_state.current_chat_id = next(iter(st.session_state.chats))
                        else:
                            yeni_sohbet_olustur()
                    sohbetleri_kaydet()
                    st.rerun()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_source_id" not in st.session_state:
    st.session_state.processed_source_id = None
if "full_text" not in st.session_state:
    st.session_state.full_text = None
if "summary" not in st.session_state:
    st.session_state.summary = None

aktif_sohbet = st.session_state.chats[st.session_state.current_chat_id]

MAX_SUMMARY_CHARS = 300000


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


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=google_api_key
)

belge_modu = st.session_state.vectorstore is not None

if belge_modu:
    col_bilgi, col_kaldir = st.columns([6, 1])
    with col_bilgi:
        st.info(t("belge_eklendi_bilgi"))
    with col_kaldir:
        if st.button(t("belge_kaldir")):
            st.session_state.vectorstore = None
            st.session_state.processed_source_id = None
            st.session_state.full_text = None
            st.session_state.summary = None
            st.rerun()

    st.markdown(t("ozet_baslik"))
    if st.button(t("ozetle_buton")):
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
        st.write(st.session_state.summary)
        if st.session_state.get("summary_truncated"):
            st.caption(t("ozet_kisaltildi"))

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

    sohbetleri_kaydet()

    if baslik_ilk_kez_degisti and user_question:
        st.rerun()
