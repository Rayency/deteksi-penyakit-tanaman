import os
import json

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="Deteksi Penyakit Daun",
    page_icon="🌿",
    layout="centered",
)

MODEL_PATH = "model/best_mobilenetv2_2phase.pth"
CLASS_PATH = "model/class_names.json"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ambang batas confidence untuk menentukan status hasil
AMBANG_TIDAK_DIKENALI = 50.0   # di bawah ini -> kemungkinan bukan salah satu spesies yang dilatih
AMBANG_KURANG_YAKIN   = 75.0   # di bawah ini -> tampilkan, tapi beri catatan kehati-hatian

# 13 spesies tanaman yang ada di dataset training (PlantDoc, 28 kelas penyakit)
SPESIES_DIDUKUNG = [
    ("AP", "Apple"), ("BP", "Bell Pepper"), ("BL", "Blueberry"),
    ("CH", "Cherry"), ("CN", "Corn"), ("PC", "Peach"),
    ("PT", "Potato"), ("RB", "Raspberry"), ("SB", "Soybean"),
    ("SQ", "Squash"), ("ST", "Strawberry"), ("TM", "Tomato"), ("GR", "Grape"),
]

# Rincian 28 kelas asli, dikelompokkan per spesies.
# sehat=True -> kelas ini representasi "tidak ada penyakit terdeteksi" untuk spesies itu
# sehat=None -> satu-satunya kategori untuk spesies ini di dataset (status tidak ditentukan)
SPESIES_KELAS = {
    "Apple":       [("Scab Leaf", False), ("Rust Leaf", False), ("Leaf", True)],
    "Bell Pepper": [("Leaf Spot", False), ("Leaf", True)],
    "Blueberry":   [("Leaf", None)],
    "Cherry":      [("Leaf", None)],
    "Corn":        [("Gray Leaf Spot", False), ("Leaf Blight", False), ("Rust Leaf", False)],
    "Peach":       [("Leaf", None)],
    "Potato":      [("Early Blight", False), ("Late Blight", False)],
    "Raspberry":   [("Leaf", None)],
    "Soybean":     [("Leaf", None)],
    "Squash":      [("Powdery Mildew", False)],
    "Strawberry":  [("Leaf", None)],
    "Tomato":      [("Early Blight", False), ("Septoria Leaf Spot", False), ("Leaf", True),
                     ("Bacterial Spot", False), ("Late Blight", False), ("Mosaic Virus", False),
                     ("Yellow Virus", False), ("Mold Leaf", False), ("Two Spotted Spider Mites", False)],
    "Grape":       [("Leaf", True), ("Black Rot", False)],
}

ICON_DAUN = """<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#1F3A24"
stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 21C12 21 4.5 16.5 4.5 9.5C4.5 5.5 7.5 3 12 3C16.5 3 19.5 5.5 19.5 9.5C19.5 16.5 12 21 12 21Z"/>
<path d="M12 21V6"/><path d="M8.2 9.3C9.6 8.3 10.6 8.3 12 8.3"/>
<path d="M15.8 12.2C14.4 11.2 13.4 11.2 12 11.2"/></svg>"""

# ============================================================
# CSS — tema "kartu spesimen lapangan"
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #F6F7F2; }
.block-container { padding-top: 3.6rem; max-width: 760px; }
[data-testid="stHeader"] { background: #F6F7F2; height: 3.2rem; }

/* header */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #6B5340; margin-bottom: 0.4rem;
}
.header-row { display:flex; align-items:center; gap:0.55rem; margin-bottom:0.1rem; }
.judul-app {
    font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700;
    color: #1F3A24; margin: 0;
}
.subjudul-app { color: #4B5A45; font-size: 0.95rem; margin: 0.5rem 0 1.6rem 0; max-width: 50ch; line-height:1.5; }

/* panel spesies */
.panel-spesies {
    border: 1px solid #D8DECF; background: #FFFFFF; border-radius: 4px;
    padding: 1rem 1.1rem 1.2rem 1.1rem; margin-bottom: 1.8rem;
}
.panel-spesies .label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6B5340; margin-bottom: 0.7rem; display:block;
}
.grid-spesies { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 0.5rem; }
.tag-spesies {
    display:flex; align-items:center; gap:0.45rem; border: 1px solid #E3E7DA;
    border-radius: 3px; padding: 0.35rem 0.5rem; background: #FAFBF8;
}
.tag-spesies .kode {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
    color: #FFFFFF; background: #2F5233; border-radius: 2px; padding: 0.08rem 0.32rem;
}
.tag-spesies .nama { font-size: 0.82rem; color:#2A3328; }

/* detail 28 kelas per spesies (di dalam expander) */
.grup-spesies { margin-bottom: 0.85rem; }
.grup-spesies:last-child { margin-bottom: 0; }
.grup-judul {
    display:flex; align-items:center; gap:0.4rem; margin-bottom: 0.35rem;
}
.grup-judul .kode {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; font-weight: 600;
    color: #FFFFFF; background: #2F5233; border-radius: 2px; padding: 0.06rem 0.3rem;
}
.grup-judul .nama-spesies { font-weight: 600; font-size: 0.85rem; color:#1F3A24; }
.daftar-kelas { font-size: 0.8rem; color: #4B5A45; padding-left: 1.55rem; line-height:1.6; }
.daftar-kelas .sehat { color: #2F5233; font-weight: 600; }

/* label foto sampel */
.label-foto {
    font-family:'IBM Plex Mono', monospace; font-size:0.72rem; letter-spacing:0.08em;
    text-transform:uppercase; color:#6B5340; margin-bottom:0.4rem;
}

/* kartu spesimen (hasil terdeteksi) */
.kartu-spesimen { border: 1px solid #1F3A24; background: #FFFFFF; padding: 1.4rem 1.5rem; position: relative; }
.kartu-spesimen::before { content: ""; position: absolute; inset: 6px; border: 1px dashed #C7CFB9; pointer-events: none; }
.spesimen-id { font-family:'IBM Plex Mono', monospace; font-size: 0.72rem; color:#6B5340; letter-spacing:0.06em; margin-bottom: 0.5rem; }
.spesimen-nama { font-family:'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #1F3A24; margin: 0.25rem 0 1.1rem 0; }

.meter-track {
    position: relative; height: 22px; margin-top: 26px;
    background-image: repeating-linear-gradient(to right, #D8DECF 0, #D8DECF 1px, transparent 1px, transparent 5%);
    background-color: #EEF1E7; border: 1px solid #D8DECF;
}
.meter-fill { position:absolute; top:0; left:0; height:100%; opacity:0.88; }
.meter-marker { position:absolute; top:-24px; transform: translateX(-50%); font-family:'IBM Plex Mono', monospace; font-size:0.78rem; font-weight:600; white-space:nowrap; }
.catatan-status { margin-top: 0.9rem; font-size: 0.82rem; color: #5A6353; line-height:1.5; }

/* kartu tidak dikenali */
.kartu-tidak-dikenali { border: 1px solid #8C3B30; border-left: 5px solid #8C3B30; background: #FBF4F2; padding: 1.2rem 1.4rem; }
.tdk-judul { font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1.15rem; color:#8C3B30; margin-bottom:0.45rem; }
.tdk-teks { font-size:0.88rem; color:#5A4540; line-height:1.55; }
.tdk-tebakan { margin-top:0.8rem; font-family:'IBM Plex Mono', monospace; font-size:0.74rem; color:#9C8076; }

.kotak-kosong { border: 1px dashed #C7CFB9; padding: 2.4rem 1rem; text-align: center; color: #6B5340; background-color: #FAFBF8; }
.kotak-kosong small { font-family:'IBM Plex Mono', monospace; }

[data-testid="stExpander"] { border: 1px solid #D8DECF !important; border-radius: 4px !important; background: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ARSITEKTUR MODEL (harus sama dengan saat training)
# ============================================================
def bangun_model(jumlah_kelas: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, jumlah_kelas)
    )
    return model


@st.cache_resource(show_spinner=False)
def muat_model():
    with open(CLASS_PATH, "r") as f:
        nama_kelas = json.load(f)

    model = bangun_model(len(nama_kelas))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, nama_kelas


transform_inferensi = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def prediksi(gambar: Image.Image, model, nama_kelas):
    img = gambar.convert("RGB")
    tensor = transform_inferensi(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    idx = int(torch.argmax(probs).item())
    label = nama_kelas[idx].replace("_", " ").title()
    confidence = float(probs[idx].item()) * 100
    return idx, label, confidence


# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="eyebrow">Alat Diagnostik Lapangan &middot; MobileNetV2</div>
<div class="header-row">{ICON_DAUN}<h1 class="judul-app">Deteksi Penyakit Daun</h1></div>
<p class="subjudul-app">Upload foto daun untuk mendeteksi jenis penyakitnya. Model ini dilatih
khusus pada 13 spesies tanaman di bawah — daun di luar daftar ini tidak akan terdeteksi
dengan akurat.</p>
""", unsafe_allow_html=True)

# ============================================================
# PANEL SPESIES YANG DIDUKUNG
# ============================================================
tag_html = "".join(
    f'<div class="tag-spesies"><span class="kode">{kode}</span><span class="nama">{nama}</span></div>'
    for kode, nama in SPESIES_DIDUKUNG
)
st.markdown(f"""
<div class="panel-spesies">
    <span class="label">13 Spesies Tanaman yang Didukung</span>
    <div class="grid-spesies">{tag_html}</div>
</div>
""", unsafe_allow_html=True)

with st.expander("Lihat detail 28 kategori penyakit per spesies"):
    kode_per_spesies = dict(SPESIES_DIDUKUNG)
    for nama_spesies, daftar_kelas in SPESIES_KELAS.items():
        kode = [k for k, n in SPESIES_DIDUKUNG if n == nama_spesies][0]
        baris = []
        for nama_kls, sehat in daftar_kelas:
            if sehat is True:
                baris.append(f'<span class="sehat">{nama_kls} (tidak ada penyakit)</span>')
            else:
                baris.append(nama_kls)
        st.markdown(f"""
        <div class="grup-spesies">
            <div class="grup-judul"><span class="kode">{kode}</span><span class="nama-spesies">{nama_spesies}</span></div>
            <div class="daftar-kelas">{" &middot; ".join(baris)}</div>
        </div>
        """, unsafe_allow_html=True)
    st.caption(
        "Catatan: spesies dengan satu kategori saja (Blueberry, Cherry, Peach, Raspberry, "
        "Soybean, Strawberry) tidak punya varian penyakit lain di dataset ini — bukan berarti "
        "spesies tersebut selalu sehat, hanya itu satu-satunya kategori yang dilatih."
    )

# ============================================================
# CEK KETERSEDIAAN MODEL
# ============================================================
if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH)):
    st.error(
        "File model tidak ditemukan.\n\n"
        f"Pastikan ada:\n- `{MODEL_PATH}`\n- `{CLASS_PATH}`"
    )
    st.stop()

model, nama_kelas = muat_model()

# ============================================================
# UPLOAD & PREDIKSI
# ============================================================
file_gambar = st.file_uploader(
    "Pilih gambar daun (.jpg / .jpeg / .png)",
    type=["jpg", "jpeg", "png"],
)

if file_gambar is not None:
    gambar = Image.open(file_gambar)

    kol_gambar, kol_hasil = st.columns([1, 1], gap="medium")

    with kol_gambar:
        st.markdown('<div class="label-foto">Foto Sampel</div>', unsafe_allow_html=True)
        st.image(gambar, use_container_width=True)

    with st.spinner("Menganalisis gambar..."):
        idx, label, confidence = prediksi(gambar, model, nama_kelas)

    with kol_hasil:
        if confidence < AMBANG_TIDAK_DIKENALI:
            st.markdown(f"""
            <div class="kartu-tidak-dikenali">
                <div class="tdk-judul">Daun Tidak Dikenali</div>
                <div class="tdk-teks">Pola visual pada gambar ini tidak cocok dengan salah satu
                dari 13 spesies tanaman yang dilatih pada model. Coba upload foto daun dari
                daftar spesies di atas.</div>
                <div class="tdk-tebakan">Tebakan terdekat (tidak diandalkan): {label} &middot; {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if confidence >= AMBANG_KURANG_YAKIN:
                warna, status = "#2F5233", "TERDETEKSI"
                catatan = ""
            else:
                warna, status = "#A6791E", "KURANG YAKIN"
                catatan = ('<div class="catatan-status">Confidence sedang — coba foto dengan '
                           'fokus lebih jelas pada area bercak/lesi untuk hasil lebih akurat.</div>')

            st.markdown(f"""
            <div class="kartu-spesimen">
                <div class="spesimen-id">SPECIMEN-{idx:02d} &middot; {status}</div>
                <div class="spesimen-nama">{label}</div>
                <div class="meter-track">
                    <div class="meter-fill" style="width:{confidence:.1f}%; background:{warna};"></div>
                    <div class="meter-marker" style="left:{confidence:.1f}%; color:{warna};">{confidence:.1f}%</div>
                </div>
                {catatan}
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="kotak-kosong">Belum ada gambar diupload<br>'
        '<small>Drag & drop atau klik tombol di atas</small></div>',
        unsafe_allow_html=True
    )

# ============================================================
# INFO MODEL
# ============================================================
with st.expander("Tentang model"):
    st.write(f"""
    - **Arsitektur:** MobileNetV2 (transfer learning, fine-tuning 2 phase)
    - **Jumlah kelas:** {len(nama_kelas)} kategori penyakit, {len(SPESIES_DIDUKUNG)} spesies tanaman
    - **Ukuran input:** {IMG_SIZE}×{IMG_SIZE} px
    - **Device:** {DEVICE.type.upper()}
    - **Ambang "tidak dikenali":** confidence < {AMBANG_TIDAK_DIKENALI:.0f}%
    """)