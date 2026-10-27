
import io
import os
import sys
import html
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tsp_core import *

MAX_CITIES = 200
FONT = "Nunito Sans, Plus Jakarta Sans, system-ui, sans-serif"

C = {
    "bg": "#FFF8F0",
    "cream": "#FFF3E2",
    "soft": "#F3DCD3",
    "pink": "#E9B1B0",
    "deep": "#B4606B",
    "gold": "#D9A441",
    "text": "#4A2C2C",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Nunito Sans", "Plus Jakarta Sans", "DejaVu Sans"
    ],
})

st.set_page_config(
    page_title="TSP Optimizer",
    page_icon="🌸",
    layout="wide",
)

# ---------- Dashboard CSS ----------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg:#FFF8F0;
    --panel:#FFFDF9;
    --peach:#FEEAC9;
    --rose:#B4606B;
    --rose-d:#9C4D58;
    --text:#4A2C2C;
    --muted:#8A6F6A;
    --line:#F0E1D2;
}

.stApp, .stApp p, .stApp label, .stApp button,
.stApp input, .stApp textarea, .stApp li, .stApp td,
.stApp th, .stApp summary, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, [data-testid=stSidebar] {
    font-family:'Nunito Sans','Plus Jakarta Sans',
    system-ui,-apple-system,'Segoe UI',sans-serif !important;
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color:var(--text);
    font-weight:700;
}

.block-container {
    padding-top:2rem;
    max-width:1250px;
}

.hero {
    display:flex;
    gap:16px;
    align-items:center;
    background:var(--peach);
    border-radius:20px;
    padding:18px 24px;
    margin-bottom:6px;
}

.hero .logo {
    width:50px;
    height:50px;
    border-radius:15px;
    background:var(--rose);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:26px;
}

.hero .t {
    font-size:1.65rem;
    font-weight:700;
    color:var(--text);
    line-height:1.2;
}

.hero .d {
    color:var(--muted);
    font-size:.95rem;
    margin-top:2px;
}

.sec {
    display:flex;
    align-items:center;
    gap:10px;
    margin:30px 0 10px;
    font-weight:700;
    font-size:1.2rem;
    color:var(--text);
}

.sec .num {
    background:var(--rose);
    color:#fff;
    border-radius:50%;
    width:27px;
    height:27px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:.85rem;
}

.panel {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:18px;
    padding:16px 20px;
    margin:10px 0;
    box-shadow:0 2px 10px rgba(120,70,60,.05);
}

.panel.draft {
    border:1.5px dashed #E3C8B8;
    background:#FFFAF3;
    box-shadow:none;
}

.ptitle {
    font-size:.76rem;
    letter-spacing:.07em;
    text-transform:uppercase;
    color:var(--muted);
    font-weight:700;
    margin-bottom:8px;
}

.mcard {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:16px;
    padding:12px 16px;
    min-height:88px;
    box-shadow:0 2px 8px rgba(120,70,60,.04);
}

.ml {
    font-size:.78rem;
    color:var(--muted);
    font-weight:600;
}

.mv {
    font-size:1.4rem;
    font-weight:700;
    color:var(--text);
    line-height:1.35;
}

.ms {
    font-size:.78rem;
    color:var(--muted);
}

.chips {
    display:flex;
    flex-wrap:wrap;
    gap:6px;
    margin:2px 0 8px;
}

.chip {
    background:var(--peach);
    color:var(--text);
    border-radius:999px;
    padding:3px 12px;
    font-size:.82rem;
    font-weight:500;
}

.chip b {
    font-weight:700;
    margin-left:6px;
}

.chip.on {
    background:var(--rose);
    color:#fff;
}

.chip.off {
    background:#F3EBE4;
    color:#A8958E;
}

.rl {
    font-size:.82rem;
    font-weight:700;
    color:var(--rose);
    margin:8px 0 2px;
}

.badge {
    display:inline-block;
    border-radius:999px;
    padding:3px 12px;
    font-size:.78rem;
    font-weight:700;
    margin-left:10px;
    vertical-align:middle;
}

.badge.todo {
    background:#FFF1D6;
    color:#9A6A12;
    border:1px dashed #E0B864;
}

.badge.done {
    background:#E6F1E4;
    color:#3F6B3A;
}

.badge.stale {
    background:#FBE3DF;
    color:#9C4D58;
}

.stButton button, .stDownloadButton button {
    border-radius:12px;
    border:1px solid var(--line);
    background:var(--peach);
    color:var(--text);
    font-weight:600;
}

.stButton button:hover, .stDownloadButton button:hover {
    border-color:var(--rose);
    color:var(--rose-d);
}

.stButton button[kind=primary],
.stButton button[data-testid=stBaseButton-primary] {
    background:var(--rose);
    color:#fff;
    border:0;
    font-size:1.05rem;
    padding:.65rem 1rem;
    box-shadow:0 4px 12px rgba(180,96,107,.3);
}

.stButton button[kind=primary]:hover,
.stButton button[data-testid=stBaseButton-primary]:hover {
    background:var(--rose-d);
    color:#fff;
}

[data-testid=stMetric] {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:16px;
    padding:12px 18px;
}

[data-testid=stSidebar] {
    border-right:1px solid var(--line);
}

button[role=tab][aria-selected=true] {
    color:var(--rose);
    font-weight:700;
}

/* Gaya khusus tab Analisis */
.analysis-heading {
    display:flex;
    align-items:center;
    gap:12px;
    padding:16px 20px;
    margin-bottom:12px;
    border:1px solid #F5DAD5;
    border-radius:16px;
    background:#FCE9E5;
    color:#B73558;
    font-size:1.2rem;
    font-weight:800;
}

.analysis-card {
    display:flex;
    align-items:flex-start;
    gap:12px;
    min-height:145px;
    height:100%;
    padding:16px;
    margin-bottom:14px;
    border:1px solid #F3DDD5;
    border-radius:16px;
    background:#FFFAF5;
    box-shadow:0 2px 8px rgba(120,70,60,.04);
}

.analysis-icon {
    flex:0 0 40px;
    height:40px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:50%;
    background:#FBE4E1;
    color:#B73558;
    font-size:1.2rem;
    font-weight:700;
}

.analysis-content {
    flex:1;
    min-width:0;
}

.analysis-title {
    color:#B73558;
    font-weight:800;
    font-size:1rem;
    margin-bottom:7px;
}

.analysis-body {
    color:#4A2C2C;
    font-size:.92rem;
    line-height:1.55;
    overflow-wrap:anywhere;
}

.analysis-note {
    display:flex;
    gap:14px;
    align-items:flex-start;
    padding:16px 20px;
    margin-top:2px;
    border:1px solid #F8DFB8;
    border-radius:16px;
    background:#FFF0D9;
    color:#604638;
    line-height:1.55;
}

.analysis-note-title {
    font-weight:800;
    margin-bottom:3px;
    color:#4A2C2C;
}
</style>
""", unsafe_allow_html=True)

ss = st.session_state

st.markdown(
    '<div class="hero"><div class="logo">🌸</div><div>'
    '<div class="t">TSP Optimizer</div>'
    '<div class="d">Optimasi rute dengan Simulated Annealing, '
    'Tabu Search, dan Variable Neighborhood Search.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)


# ---------- Reusable UI helpers ----------

def card(label, value, sub=""):
    return (
        f'<div class="mcard"><div class="ml">{label}</div>'
        f'<div class="mv">{value}</div>'
        f'<div class="ms">{sub}</div></div>'
    )


def chips(items):
    return '<div class="chips">' + "".join(
        f'<span class="chip {cls}">{key}'
        f'{"<b>" + str(value) + "</b>" if value != "" else ""}'
        '</span>'
        for key, value, cls in items
    ) + "</div>"


def sec(number, title, badge=""):
    st.markdown(
        f'<div class="sec"><span class="num">{number}</span>'
        f'{title}{badge}</div>',
        unsafe_allow_html=True,
    )


def on(value):
    return "on" if value else "off"


# ================= SIDEBAR =================

sb = st.sidebar
sb.markdown("### 🌸 Pengaturan dasar")
sb.markdown("**Data kota**")

method = sb.radio(
    "Metode input data",
    [
        "Klik kanvas X-Y",
        "Generate acak",
        "Input manual",
        "Upload Excel/CSV",
        "Dataset 6 Kota",
    ],
    label_visibility="collapsed",
)

df = None

if method == "Klik kanvas X-Y":
    cw = sb.number_input("Lebar kanvas (X max)", 10, 10000, 100)
    ch = sb.number_input("Tinggi kanvas (Y max)", 10, 10000, 100)

elif method == "Generate acak":
    n_rand = sb.number_input("Jumlah kota", 3, MAX_CITIES, 20)
    xr = sb.number_input("X max", 10, 10000, 100)
    yr = sb.number_input("Y max", 10, 10000, 100)
    sd = sb.number_input("Random seed data", 0, 99999, 42)

    rng_data = np.random.default_rng(sd)
    df = pd.DataFrame({
        "City": range(1, n_rand + 1),
        "X": rng_data.uniform(0, xr, n_rand).round(2),
        "Y": rng_data.uniform(0, yr, n_rand).round(2),
    })

elif method == "Input manual":
    sb.caption(
        "Edit tabel koordinat di halaman utama "
        "(tambah, ubah, atau hapus baris)."
    )

elif method == "Upload Excel/CSV":
    up = sb.file_uploader("File", type=["xlsx", "xls", "csv"])



    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                raw = pd.read_csv(up)
            else:
                xl = pd.ExcelFile(up)
                sheet = sb.selectbox("Sheet", xl.sheet_names)
                raw = xl.parse(sheet)

            cols = list(raw.columns)
            numeric_cols = [
                col for col in cols
                if pd.api.types.is_numeric_dtype(raw[col])
            ]

            if len(numeric_cols) < 2:
                st.error("File harus memiliki minimal dua kolom numerik.")
                st.stop()

            city_col = sb.selectbox(
                "Kolom City/ID",
                ["(otomatis 1..n)"] + cols,
            )
            col_x = sb.selectbox("Kolom X", numeric_cols, index=0)
            col_y = sb.selectbox(
                "Kolom Y",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
            )

            if col_x == col_y:
                st.error("Kolom X dan Y harus berbeda.")
                st.stop()

            df = raw[[col_x, col_y]].copy()
            df.columns = ["X", "Y"]

            if city_col in cols:
                df.insert(0, "City", raw[city_col])
            else:
                df.insert(0, "City", range(1, len(df) + 1))

            df = df.dropna(subset=["X", "Y"]).reset_index(drop=True)

        except Exception as exc:
            st.error(f"Gagal membaca file: {exc}")
            st.stop()

else:
    # Dataset contoh: 6 kota tetap (tidak lagi membaca data1.xlsx).
    sample_data = {
        "City": [1, 2, 3, 4, 5, 6],
        "X": [2.0, 652.0, 99.0, 681.0, 501.0, 188.0],
        "Y": [33.0, 32.0, 368.0, 195.0, 106.0, 664.0],
    }
    df = pd.DataFrame(sample_data)
    sb.caption("Dataset contoh: 6 kota (ilustrasi Tabu Search).")

sb.divider()
sb.markdown("**Algoritma**")
sb.markdown(
    chips([
        ("Simulated Annealing", "", "on"),
        ("Tabu Search", "", "on"),
        ("VNS", "", "on"),
    ]),
    unsafe_allow_html=True,
)



# ================= 1. DATA KOTA =================

sec(1, "Data kota")


def render_canvas(width, height):
    ss.setdefault("clicks", [])
    ss.setdefault("cv", 0)

    step = max(width, height) / 50

    st.caption(
        "Klik area untuk menambah kota. Pilih Hapus kota "
        "untuk menghapus kota terdekat."
    )

    ca, cb, cc = st.columns([2, 1, 1])
    action = ca.radio(
        "Aksi klik",
        ["Tambah kota", "Hapus kota"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if cb.button("↩️ Undo"):
        ss.clicks = ss.clicks[:-1]
        ss.cv += 1
        st.rerun()

    if cc.button("🗑️ Reset"):
        ss.clicks = []
        ss.cv += 1
        st.rerun()

    gx, gy = np.meshgrid(
        np.arange(0, width + 1e-9, width / 50),
        np.arange(0, height + 1e-9, height / 50),
    )

    fig = go.Figure(
        go.Scatter(
            x=gx.ravel(),
            y=gy.ravel(),
            mode="markers",
            marker=dict(size=13, color=C["pink"], opacity=0.05),
            hovertemplate="x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
            showlegend=False,
        )
    )

    if ss.clicks:
        points = np.array(ss.clicks)
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers+text",
                text=[str(i + 1) for i in range(len(points))],
                textposition="top center",
                marker=dict(
                    size=14,
                    color=C["pink"],
                    line=dict(color=C["deep"], width=2),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=C["cream"],
        paper_bgcolor=C["bg"],
        dragmode=False,
        font=dict(family=FONT, color=C["text"]),
        xaxis=dict(
            range=[-width * 0.02, width * 1.02],
            gridcolor=C["soft"],
        ),
        yaxis=dict(
            range=[-height * 0.02, height * 1.02],
            gridcolor=C["soft"],
        ),
    )

    event = st.plotly_chart(
        fig,
        key=f"cv{ss.cv}",
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": False},
    )

    selected = (
        (event or {}).get("selection", {}).get("points", [])
        if event else []
    )

    if selected:
        x = float(selected[0]["x"])
        y = float(selected[0]["y"])

        if action == "Tambah kota":
            duplicate = any(
                abs(x - px) + abs(y - py) < step * 0.5
                for px, py in ss.clicks
            )

            if not duplicate and len(ss.clicks) < MAX_CITIES:
                ss.clicks.append((round(x, 2), round(y, 2)))

        elif ss.clicks:
            distances = [
                abs(x - px) + abs(y - py)
                for px, py in ss.clicks
            ]
            idx = int(np.argmin(distances))

            if distances[idx] < step * 3:
                ss.clicks.pop(idx)

        ss.cv += 1
        st.rerun()

    return pd.DataFrame({
        "City": range(1, len(ss.clicks) + 1),
        "X": [point[0] for point in ss.clicks],
        "Y": [point[1] for point in ss.clicks],
    })


if method == "Klik kanvas X-Y":
    df = render_canvas(cw, ch)

elif method == "Input manual":
    base = pd.DataFrame({
        "City": [1, 2, 3, 4, 5, 6],
        "X": [10., 30, 45, 60, 20, 50],
        "Y": [25., 15, 40, 10, 50, 45],
    })

    df = st.data_editor(
        base,
        num_rows="dynamic",
        width="stretch",
        key="manual",
    )


if df is None:
    if method == "Upload Excel/CSV":
        st.info(
            "### Format file yang diperlukan\n\n"
            "Unggah file melalui sidebar di sebelah kiri.\n\n"
            "**Format:** `.xlsx`, `.xls`, atau `.csv`.\n\n"
            "Satu baris mewakili satu kota. File harus memiliki "
            "minimal dua kolom numerik berbeda untuk koordinat X dan Y.\n\n"
            "**Contoh tabel:**\n\n"
            "| City | X | Y |\n"
            "|---|---:|---:|\n"
            "| A | 10 | 25 |\n"
            "| B | 30 | 15 |\n"
            "| C | 45 | 40 |\n\n"
            "Nama kolom bebas. Pilih kolom X dan Y setelah upload. "
            "Kolom ID kota opsional. Baris pertama sebaiknya berisi "
            "nama kolom."
        )
    else:
        st.info("Lengkapi input data di sidebar terlebih dahulu.")

    st.stop()


df = df.dropna(subset=["X", "Y"]).reset_index(drop=True)

errors = []

if len(df) < 3:
    errors.append("Jumlah kota minimal 3.")

if len(df) > MAX_CITIES:
    errors.append(f"Jumlah kota maksimal {MAX_CITIES}.")

if df["City"].duplicated().any():
    errors.append("ID kota duplikat.")

if not (
    pd.api.types.is_numeric_dtype(df["X"])
    and pd.api.types.is_numeric_dtype(df["Y"])
):
    errors.append("Koordinat harus numerik.")

if errors:
    st.warning(" ".join(errors))
    st.stop()

n = len(df)
P = df[["X", "Y"]].astype(float).values
names = df["City"].astype(str).tolist()


# ---------- Plot helpers ----------

def new_ax(width=5.5, height=4.5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["cream"])
    ax.grid(color=C["soft"], linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_color(C["soft"])

    ax.tick_params(colors=C["text"])
    return fig, ax


def draw_cities(ax, route, labels=True):
    ax.scatter(
        P[:, 0], P[:, 1],
        s=55,
        c=C["pink"],
        edgecolors=C["deep"],
        zorder=3,
    )

    ax.scatter(
        *P[route[0]],
        s=170,
        marker="*",
        c=C["gold"],
        edgecolors=C["deep"],
        zorder=4,
    )

    if labels:
        for i in range(n):
            ax.annotate(
                names[i],
                P[i],
                fontsize=7,
                color=C["text"],
                xytext=(4, 4),
                textcoords="offset points",
            )


def fig_route(route, title, labels=True):
    fig, ax = new_ax()
    closed_route = list(route) + [route[0]]

    ax.plot(
        P[closed_route, 0],
        P[closed_route, 1],
        "-",
        color=C["deep"],
        linewidth=1.6,
        zorder=2,
    )

    draw_cities(ax, route, labels)
    ax.set_title(
        f"{title}\nJarak = {length(route, D):.2f}",
        color=C["text"],
    )

    return fig


def edges(route):
    return {
        frozenset((route[i], route[(i + 1) % len(route)]))
        for i in range(len(route))
    }


def fig_diff(route, other, title, color):
    fig, ax = new_ax()
    shared = edges(route) & edges(other)

    for i in range(n):
        u = route[i]
        v = route[(i + 1) % n]
        is_shared = frozenset((u, v)) in shared

        ax.plot(
            P[[u, v], 0],
            P[[u, v], 1],
            color=C["soft"] if is_shared else color,
            linewidth=2 if is_shared else 3,
            zorder=2,
        )

    draw_cities(ax, route, False)
    ax.set_title(
        f"{title}: {length(route, D):.2f}",
        color=C["text"],
    )

    return fig, len(shared)


def png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode()


if method != "Klik kanvas X-Y":
    col_data, col_plot = st.columns([1, 2])
    col_data.dataframe(
        df,
        height=300,
        width="stretch",
        hide_index=True,
    )

    f0, a0 = new_ax()
    a0.scatter(
        P[:, 0], P[:, 1],
        s=55,
        c=C["pink"],
        edgecolors=C["deep"],
    )

    for i in range(n):
        a0.annotate(
            names[i],
            P[i],
            fontsize=7,
            color=C["text"],
            xytext=(4, 4),
            textcoords="offset points",
        )

    a0.set_title("Sebaran kota", color=C["text"])
    col_plot.pyplot(f0)
    plt.close(f0)

else:
    with st.expander(f"Tabel koordinat ({n} kota)"):
        st.dataframe(df, width="stretch", hide_index=True)


# ================= 2. PENGATURAN ALGORITMA =================

sec(2, "Pengaturan algoritma")

with st.expander(
    "⚙️ Pengaturan lanjutan (klik untuk membuka)",
    expanded=False,
):
    tab_names = [
        "Jarak & Solusi Awal",
        "Local Search",
        "Simulated Annealing",
        "Tabu Search",
        "Variable Neighborhood Search (VNS)",
    ]


    tab_names.append("Multi-run & Seed")
    setting_tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with setting_tabs["Jarak & Solusi Awal"]:
        a, b = st.columns(2)

        metric = a.selectbox(
            "Distance metric",
            ["Euclidean", "Manhattan"],
        )

        init_mode = a.radio(
            "Rute awal",
            ["Otomatis (heuristik)", "Manual (urutan dari user)"],
        )

        manual_ok = init_mode.startswith("Manual")

        if manual_ok:
            # Mode manual: kota pertama pada rute menjadi titik awal.
            start_city = None

            manual_txt = b.text_input(
                "Rute manual (mis. 1,4,2,5,3,6)",
                "",
                help=(
                    "Masukkan ID setiap kota tepat satu kali, dipisahkan koma. "
                    "Kota pertama menjadi titik awal dan titik akhir."
                ),
            )

            cons = "NN"

        else:
            # Mode otomatis: pengguna memilih kota awal dan heuristik.
            start_city = b.number_input(
                "Kota awal (nomor baris, mulai 1)",
                min_value=1,
                max_value=MAX_CITIES,
                value=1,
            )

            manual_txt = ""

            cons = b.selectbox(
                "Construction heuristic",
                [
                    "NN",
                    "NN with Variation",
                    "NI",
                    "CI",
                    "FI",
                    "AI",
                ],
                help=(
                    "NN: nearest neighbor; "
                    "NN with Variation: extend either end of the open path; "
                    "NI: nearest insertion; CI: cheapest insertion; "
                    "FI: farthest insertion; AI: arbitrary insertion"
                ),
            )


    with setting_tabs["Local Search"]:
        a, b = st.columns(2)

        use_2opt = a.checkbox("2-opt pada solusi awal", True)
        ls_after = a.checkbox(
            "2-opt setelah SA/TS/VNS (polish)",
            False,
        )
        ls_strategy = b.selectbox(
            "Improvement strategy",
            ["Best Improvement", "First Improvement"],
        )

        st.caption(
            "Operator SA dan TS dipilih pada pengaturan masing-masing. "
            "VNS menggunakan neighborhood Swap, Insertion, dan 2-opt."
        )

    with setting_tabs["Simulated Annealing"]:
        a, b, c = st.columns(3)

        sa_T0 = a.number_input(
            "Initial temperature", 0.1, 1e7, 1000.0
        )
        sa_alpha = b.number_input(
            "Alpha (cooling rate)",
            0.5, 0.9999, 0.995, 0.001,
            format="%.4f",
        )
        sa_Tmin = c.number_input(
            "Minimum temperature", 1e-6, 1e5, 0.1,
            format="%.4f",
        )
        shared_max_iter = b.number_input("Maximum iteration (SA & TS bersama)", 10, 1000000, 3000)
        sa_per = 1  # kompatibilitas internal; SA mencoba satu kandidat per langkah
        sa_max = int(shared_max_iter)
        sa_op = c.selectbox("Neighborhood SA", ["2-opt", "Swap", "3-opt"])

    with setting_tabs["Tabu Search"]:
        a, b, c = st.columns(3)

        ts_max = int(shared_max_iter)
        a.caption(f"Batas iterasi bersama SA & TS: {int(shared_max_iter)}")
        ts_size = b.number_input("Tabu size", 1, 1000, 10)
        ts_nbr = c.number_input("Neighborhood size", 3, 5000, 50)

        ts_op = a.selectbox("Neighborhood TS", ["2-opt", "Swap", "3-opt"])
        ts_asp = b.checkbox("Aspiration criterion", True)

    with setting_tabs["Variable Neighborhood Search (VNS)"]:
        a, b, c = st.columns(3)

        vns_max = a.number_input(
            "Maximum iteration VNS",
            min_value=0,
            max_value=100000,
            value=500,
        )

        vns_kmax = b.number_input(
            "Jumlah maksimum neighborhood (k_max)",
            min_value=1,
            max_value=3,
            value=3,
        )

        vns_neighborhoods = c.multiselect(
            "Struktur neighborhood",
            ["Swap", "Insertion", "2-opt"],
            default=["Swap", "Insertion", "2-opt"],
        )

        a, b = st.columns(2)

        vns_shaking = a.selectbox(
            "Metode shaking",
            ["Satu move acak", "Beberapa move (k+1)"],
        )

        vns_local_search = b.checkbox(
            "Gunakan local search 2-opt setelah shaking",
            True,
        )


    with setting_tabs["Multi-run & Seed"]:
        a, b = st.columns(2)

        runs = a.number_input(
            "Multi-run (jumlah percobaan tiap metode)",
            1, 50, 1,
        )

        seed = b.number_input(
            "Random seed algoritma",
            0, 99999, 1,
        )

if not vns_neighborhoods:
    st.warning("Pilih minimal satu neighborhood untuk VNS.")

D = dist_matrix(P, metric)


# ================= 3. SOLUSI AWAL DAN RINGKASAN =================


def build_init():
    if manual_ok:
        ids = [item.strip() for item in manual_txt.split(",")]

        if any(not item for item in ids):
            raise ValueError(
                "Rute manual tidak boleh memiliki ID kota kosong."
            )

        try:
            route = [names.index(item) for item in ids]
        except ValueError:
            raise ValueError(
                "Rute manual berisi ID kota yang tidak ada pada dataset."
            )

        if len(route) != n or sorted(route) != list(range(n)):
            raise ValueError(
                "Rute manual harus memuat setiap kota tepat satu kali."
            )

        return route

    # Bagian ini hanya dijalankan pada mode otomatis.
    start = int(start_city) - 1

    if start >= n:
        raise ValueError(
            f"Kota awal ({int(start_city)}) melebihi jumlah kota ({n})."
        )

    rng_init = np.random.default_rng(int(seed))

    if cons == "NN":
        return nearest_neighbor(D, start)

    if cons == "NN with Variation":
        return nearest_neighbor_variation(D, start)

    return insertion(D, start, cons, rng_init)



try:
    init_preview = build_init()
    init_err = None
except ValueError as exc:
    init_preview = None
    init_err = str(exc)

cfg = dict(
    method=method,
    n=n,
    metric=metric,
    init_mode=init_mode,
    cons=cons,
    start_city=int(start_city) if not manual_ok else None,
    manual_txt=manual_txt,
    seed=int(seed),
    use_2opt=use_2opt,
    ls_strategy=ls_strategy,
    ls_after=ls_after,
    sa=(
        sa_T0, sa_alpha, sa_Tmin,
        int(sa_per), int(sa_max), sa_op,
    ),
    ts=(
        int(ts_max), int(ts_size), int(ts_nbr),
        ts_op, ts_asp,
    ),
    vns=(
        int(vns_max),
        int(vns_kmax),
        tuple(vns_neighborhoods),
        vns_shaking,
        bool(vns_local_search),
    ),
    runs=int(runs),
)

out = ss.get("out")

# Hapus hasil lama yang belum memiliki integrasi VNS.
if out and (
    "VNS" not in out.get("R", {})
    or "vns" not in out.get("cfg", {})
):
    ss.pop("out", None)
    out = None

ran = (
    out is not None
    and out["n"] == n
    and np.array_equal(out["P"], P)
    and out["metric"] == metric
)

badge = (
    '<span class="badge todo">Konfigurasi · belum dijalankan</span>'
    if not ran else
    '<span class="badge done">Sudah dijalankan · konfigurasi sama</span>'
    if out["cfg"] == cfg else
    '<span class="badge stale">Konfigurasi berubah sejak run terakhir</span>'
)

sec(3, "Ringkasan konfigurasi", badge)

st.caption(
    "Ini adalah ringkasan pengaturan yang akan dijalankan. "
    "Hasil optimasi muncul setelah RUN OPTIMIZATION ditekan."
)

st.markdown(
    '<div class="ptitle">A · Problem overview</div>',
    unsafe_allow_html=True,
)

overview_cols = st.columns(4)

overview = [
    ("Jumlah kota", n, "node"),
    ("Distance metric", metric, ""),
    ("Metode input", method.split(" (")[0], ""),
    (
        "Solusi awal",
        "Manual" if manual_ok else cons,
        "rute dari user" if manual_ok else "construction heuristic",
    ),
]

for col, (label, value, sub) in zip(overview_cols, overview):
    col.markdown(card(label, value, sub), unsafe_allow_html=True)

st.markdown(
    '<div class="panel"><div class="ptitle">'
    'B · Optimization configuration</div>'
    + chips([
        ("Simulated Annealing", "", "on"),
        ("Tabu Search", "", "on"),
        ("VNS", "", "on"),
        ("2-opt awal", "ON" if use_2opt else "OFF", on(use_2opt)),
        ("Polish", "ON" if ls_after else "OFF", on(ls_after)),
        (ls_strategy, "", ""),
        ("Multi-run", f"{int(runs)}×", on(runs > 1)),
        ("Seed", int(seed), ""),
    ])
    + '<div class="rl">Simulated Annealing</div>'
    + chips([
        ("T0", sa_T0, ""),
        ("α", sa_alpha, ""),
        ("Tmin", sa_Tmin, ""),
        ("iter/T", int(sa_per), ""),
        ("maxIter", int(sa_max), ""),
        ("operator", sa_op, ""),
    ])
    + '<div class="rl">Tabu Search</div>'
    + chips([
        ("maxIter", int(ts_max), ""),
        ("tabu size", int(ts_size), ""),
        ("neighborhood", int(ts_nbr), ""),
        ("operator", ts_op, ""),
        ("aspiration", "ON" if ts_asp else "OFF", on(ts_asp)),
    ])
    + '<div class="rl">Variable Neighborhood Search</div>'
    + chips([
        ("maxIter", int(vns_max), ""),
        ("k_max", int(vns_kmax), ""),
        ("neighborhoods", ", ".join(vns_neighborhoods), ""),
        ("shaking", vns_shaking, ""),
        ("local search", "ON" if vns_local_search else "OFF", on(vns_local_search)),
    ])
    + (
    )
    + "</div>",
    unsafe_allow_html=True,
)

if init_err:
    st.error(init_err)

else:
    initial_distance = length(init_preview, D)
    route_preview = init_preview[:14]

    route_chips = chips(
        [(html.escape(names[i]), "", "") for i in route_preview]
        + (
            [(f"… +{n - 14} kota", "", "off")]
            if n > 14 else []
        )
    )

    initial_cols = st.columns([1, 1, 3])

    initial_cols[0].markdown(
        card("Jarak rute awal", f"{initial_distance:.2f}", "sebelum optimasi"),
        unsafe_allow_html=True,
    )

    initial_cols[1].markdown(
        card(
            "Kota awal",
            html.escape(names[init_preview[0]]),
            "titik mulai dan akhir",
        ),
        unsafe_allow_html=True,
    )

    initial_cols[2].markdown(
        '<div class="mcard"><div class="ml">C · Urutan kota awal</div>'
        + route_chips
        + "</div>",
        unsafe_allow_html=True,
    )

    if n > 14:
        with st.expander("Lihat seluruh rute awal"):
            st.write(" → ".join(
                names[i] for i in init_preview + [init_preview[0]]
            ))


# ================= RUN OPTIMIZATION =================

st.write("")
run_cols = st.columns([2, 3])

go_run = run_cols[0].button(
    "▶ RUN OPTIMIZATION",
    type="primary",
    width="stretch",
    disabled=(
        init_err is not None
        or not vns_neighborhoods
    ),
)

run_cols[1].caption(
    f"Menjalankan SA, Tabu Search, dan VNS pada {n} kota"
   
    + (
        f", masing-masing {int(runs)} kali."
        if runs > 1 else "."
    )
)

if go_run:
    init = init_preview
    start = two_opt_ls(init, D, ls_strategy) if use_2opt else init

    def polish(result):
        if not ls_after:
            return result

        route = two_opt_ls(result["route"], D, ls_strategy)

        return {
            **result,
            "route": route,
            "dist": length(route, D),
        }

    def run_sa(run_seed):
        return polish(simulated_annealing(
            start, D,
            sa_T0, sa_alpha, sa_Tmin,
            int(sa_per), int(sa_max),
            sa_op, run_seed,
        ))

    def run_ts(run_seed):
        return polish(tabu_search(
            start, D,
            int(ts_max), int(ts_size), int(ts_nbr),
            ts_op, ts_asp, run_seed,
            detail=(n <= 12),
        ))

    def run_vns(run_seed):
        return polish(variable_neighborhood_search(
            start,
            D,
            max_iter=int(vns_max),
            k_max=int(vns_kmax),
            neighborhoods=vns_neighborhoods,
            shaking=vns_shaking,
            local_search=vns_local_search,
            strategy=ls_strategy,
            seed=run_seed,
        ))

    with st.spinner("Menjalankan optimasi SA, TS, dan VNS..."):
        R = {
            "SA": run_sa(int(seed)),
            "Tabu Search": run_ts(int(seed)),
            "VNS": run_vns(int(seed)),
        }

        multi = None

        if runs > 1:
            runners = [
                ("SA", run_sa),
                ("Tabu Search", run_ts),
                ("VNS", run_vns),
            ]

            multi = {
                name: [
                    function(int(seed) + i)
                    for i in range(int(runs))
                ]
                for name, function in runners
            }

    ss["out"] = {
        "init": init,
        "start": start,
        "R": R,
        "multi": multi,
        "P": P.copy(),
        "D": D.copy(),
        "names": names[:],
        "metric": metric,
        "n": n,
        "cfg": cfg.copy(),
    }

    st.rerun()


# ================= 4. HASIL OPTIMASI =================

out = ss.get("out")

if (
    out is None
    or out["n"] != n
    or not np.array_equal(out["P"], P)
    or out["metric"] != metric
    or "VNS" not in out["R"]
):
    sec(
        4,
        "Hasil optimasi",
        '<span class="badge todo">belum ada hasil</span>',
    )
    st.markdown(
        '<div class="panel draft">Belum ada hasil untuk data dan '
        'konfigurasi saat ini. Atur konfigurasi, kemudian tekan '
        '<b>RUN OPTIMIZATION</b>.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

result_cfg = out["cfg"]
stale = result_cfg != cfg

sec(
    4,
    "Hasil optimasi",
    (
        '<span class="badge stale">konfigurasi lama</span>'
        if stale else
        '<span class="badge done">sesuai konfigurasi</span>'
    ),
)

if stale:
    st.warning(
        "Pengaturan berubah sejak run terakhir. Hasil berikut memakai "
        "konfigurasi sebelumnya; tekan RUN untuk memperbarui."
    )

use_2opt_result = result_cfg["use_2opt"]
cons_result = result_cfg["cons"]
init_mode_result = result_cfg["init_mode"]
ts_size_result = result_cfg["ts"]
runs_result = result_cfg["runs"]

D = out["D"]
init = out["init"]
start = out["start"]
R = out["R"]
multi = out["multi"]

d0 = length(init, D)
ds = length(start, D)

sa = R["SA"]
ts = R["Tabu Search"]
vns = R["VNS"]
best_d = min(result["dist"] for result in R.values())

routes = {
    "Initial": init,
    **({"Initial + 2-opt": start} if use_2opt_result else {}),
    **{name: result["route"] for name, result in R.items()},
}

labels = [
    "📊 Ringkasan",
    "🗺️ Rute",
    "⚔️ Perbandingan",
    "📈 Konvergensi",
    "📋 Detail & Log",
    "🧠 Analisis",
]

if n <= 12:
    labels.append("🔍 Tabu Explorer")

if multi:
    labels.append("🎲 Multi-run")

T = dict(zip(labels, st.tabs(labels)))


# ---------- Ringkasan ----------

with T["📊 Ringkasan"]:
    c1, c2, c3 = st.columns(3)

    c1.metric("Jarak awal (heuristik)", f"{d0:.2f}")
    c2.metric("Jarak terbaik", f"{best_d:.2f}")

    initial_improvement = (
        100 * (d0 - best_d) / d0 if d0 > 1e-12 else 0.0
    )

    c3.metric("Peningkatan", f"{initial_improvement:.2f}%")

    rows = [{
        "Metode": "Initial" + (" + 2-opt" if use_2opt_result else ""),
        "Jarak Awal": d0,
        "Jarak Akhir": ds,
        "Improvement (%)": (
            100 * (d0 - ds) / d0 if d0 > 1e-12 else 0.0
        ),
        "Runtime (s)": 0.0,
        "Iterasi": 0,
    }]

    for name, result in R.items():
        improvement = (
            100 * (ds - result["dist"]) / ds
            if ds > 1e-12 else 0.0
        )

        rows.append({
            "Metode": name,
            "Jarak Awal": ds,
            "Jarak Akhir": result["dist"],
            "Improvement (%)": improvement,
            "Runtime (s)": result["time"],
            "Iterasi": result["iters"],
        })

    st.dataframe(
        pd.DataFrame(rows).round(4),
        width="stretch",
        hide_index=True,
    )


# ---------- Visualisasi rute ----------

with T["🗺️ Rute"]:
    show_labels = st.checkbox("Tampilkan label kota", True)
    route_tabs = st.tabs(list(routes))

    for tab, (name, route) in zip(route_tabs, routes.items()):
        with tab:
            fig = fig_route(route, name, show_labels)
            st.pyplot(fig)
            st.download_button(
                "💾 Save PNG",
                png(fig),
                f"route_{name}.png",
                "image/png",
                key=f"route_png_{name}",
            )


# ---------- Perbandingan algoritma ----------

with T["⚔️ Perbandingan"]:
    st.caption(
        "Garis berwarna menunjukkan edge yang berbeda. "
        "Garis pink muda menunjukkan edge yang sama."
    )

    pair_list = [
        ("SA", "Tabu Search", C["deep"]),
        ("SA", "VNS", C["gold"]),
        ("Tabu Search", "VNS", "#B5838D"),
    ]

    pair_cols = st.columns(3)

    for col, (name_a, name_b, color) in zip(pair_cols, pair_list):
        fig, shared_count = fig_diff(
            R[name_a]["route"],
            R[name_b]["route"],
            f"{name_a} vs {name_b}",
            color,
        )

        with col:
            st.pyplot(fig)

            st.caption(
                f"Edge yang sama: {shared_count} dari {n}. "
                f"Selisih jarak: "
                f"{abs(R[name_a]['dist'] - R[name_b]['dist']):.2f}."
            )

            st.download_button(
                f"💾 Save {name_a} vs {name_b}",
                png(fig),
                f"comparison_{name_a}_{name_b}.png",
                "image/png",
                key=f"comparison_{name_a}_{name_b}",
            )


# ---------- Konvergensi ----------

with T["📈 Konvergensi"]:
    fig, ax = new_ax(8, 3.8)

    palette = [C["deep"], C["gold"], "#B5838D", "#8C6BA8"]

    for (name, result), color in zip(R.items(), palette):
        ax.plot(
            result["hist"],
            label=name,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Best distance")
    ax.legend()

    st.pyplot(fig)

    st.download_button(
        "💾 Save PNG konvergensi",
        png(fig),
        "convergence.png",
        "image/png",
    )


# ---------- Detail dan log ----------

with T["📋 Detail & Log"]:
    selected_route = st.selectbox("Pilih rute", list(routes))
    route = routes[selected_route]

    detail_df = pd.DataFrame({
        "Step": range(1, n + 1),
        "From": [
            names[route[i]] for i in range(n)
        ],
        "To": [
            names[route[(i + 1) % n]] for i in range(n)
        ],
        "Distance": [
            round(D[route[i], route[(i + 1) % n]], 4)
            for i in range(n)
        ],
    })

    st.dataframe(detail_df, width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Export rute CSV",
        csv_bytes(detail_df),
        f"route_{selected_route}.csv",
        "text/csv",
    )

    for name, result in R.items():
        log_df = pd.DataFrame(result["log"]).drop(
            columns=[
                "Candidates",
                "Current_Before",
                "Current_After",
            ],
            errors="ignore",
        )

        with st.expander(f"Iteration log {name}"):
            st.dataframe(log_df, width="stretch")

        st.download_button(
            f"⬇️ Export iteration log {name}",
            csv_bytes(log_df),
            f"log_{name}.csv",
            "text/csv",
            key=f"log_{name}",
        )

    distance_df = pd.DataFrame(
        D, index=names, columns=names
    ).round(4)

    with st.expander("Distance matrix"):
        st.dataframe(distance_df, width="stretch")

    st.download_button(
        "⬇️ Export distance matrix",
        distance_df.to_csv().encode(),
        "distance_matrix.csv",
        "text/csv",
    )


# ---------- Analisis hasil: enam kartu ----------

with T["🧠 Analisis"]:
    eps = 1e-9

    def convergence_iteration(result):
        history = result.get("hist", [])
        return next(
            (
                i for i, value in enumerate(history)
                if value <= result["dist"] + eps
            ),
            max(0, len(history) - 1),
        )

    def improvement_text(result):
        if ds <= eps:
            return "tidak terdefinisi karena jarak awal 0"
        pct = 100 * (ds - result["dist"]) / ds
        return f"{pct:.2f}% dari solusi awal"

    def shared_edge_count(name_a, name_b):
        return len(
            edges(R[name_a]["route"])
            & edges(R[name_b]["route"])
        )

    def analysis_card(icon, title, body):
        return f"""
        <div class="analysis-card">
            <div class="analysis-icon">{icon}</div>
            <div class="analysis-content">
                <div class="analysis-title">{title}</div>
                <div class="analysis-body">{body}</div>
            </div>
        </div>
        """

    st.markdown(
        '<div class="analysis-heading">'
        '<span>💡</span> Analisis hasil</div>',
        unsafe_allow_html=True,
    )

    # Kartu 1 — Solusi awal
    initial_text = (
        f"Solusi awal "
        f"({'manual' if init_mode_result.startswith('Manual') else html.escape(str(cons_result))}) "
        f"berjarak <b>{d0:.2f}</b>"
    )

    if use_2opt_result:
        initial_text += f"; setelah 2-opt menjadi <b>{ds:.2f}</b>."
    else:
        initial_text += "."

    # Kartu 2 — Hasil SA, TS, dan VNS
    result_text = "; ".join(
        f"<b>{html.escape(name)}</b> menghasilkan "
        f"<b>{result['dist']:.2f}</b> "
        f"({improvement_text(result)})"
        for name, result in R.items()
    ) + "."

    # Kartu 3 — Perbandingan hasil aktual
    comparisons = []

    for name_a, name_b in combinations(R.keys(), 2):
        distance_a = R[name_a]["dist"]
        distance_b = R[name_b]["dist"]

        if abs(distance_a - distance_b) <= eps:
            comparisons.append(
                f"{html.escape(name_a)} dan {html.escape(name_b)} "
                "menghasilkan jarak yang sama"
            )
        else:
            better = name_a if distance_a < distance_b else name_b
            other = name_b if better == name_a else name_a
            gap = abs(distance_a - distance_b)

            comparisons.append(
                f"{html.escape(better)} menghasilkan jarak "
                f"{gap:.2f} lebih pendek dibandingkan "
                f"{html.escape(other)}"
            )

    best_names = [
        name for name, result in R.items()
        if abs(result["dist"] - best_d) <= eps
    ]

    comparison_text = (
        "; ".join(comparisons)
        + f". Jarak terbaik <b>{best_d:.2f}</b> dihasilkan oleh "
        + f"<b>{', '.join(html.escape(x) for x in best_names)}</b>."
    )

    # Kartu 4 — Waktu komputasi
    runtime_text = "; ".join(
        f"<b>{html.escape(name)}</b>: {result['time']:.3f} s"
        for name, result in R.items()
    )

    fastest_time = min(result["time"] for result in R.values())
    fastest_names = [
        name for name, result in R.items()
        if abs(result["time"] - fastest_time) <= eps
    ]

    runtime_text += (
        ". Runtime tercepat: "
        f"<b>{', '.join(html.escape(x) for x in fastest_names)}</b>."
    )

    # Kartu 5 — Iterasi saat best distance final tercapai
    iteration_text = "; ".join(
        f"<b>{html.escape(name)}</b>: best distance final pertama kali "
        f"tercapai pada iterasi {convergence_iteration(result)} "
        f"dari {result['iters']} iterasi"
        for name, result in R.items()
    ) + "."

    # Kartu 6 — Kesamaan edge antar-rute
    edge_text = "; ".join(
        f"{html.escape(name_a)}–{html.escape(name_b)}: "
        f"{shared_edge_count(name_a, name_b)} dari {n} edge sama"
        for name_a, name_b in combinations(R.keys(), 2)
    ) + "."

    cards = [
        ("↗", "Solusi awal", initial_text),
        ("▥", "Hasil SA, TS, dan VNS", result_text),
        ("⚖", "Perbandingan hasil", comparison_text),
        ("◷", "Waktu komputasi", runtime_text),
        ("↻", "Perbaikan solusi", iteration_text),
        ("⌘", "Struktur rute", edge_text),
    ]

    # Susunan tiga kolom dan dua baris
    for row_index in range(2):
        cols = st.columns(3)
        row_cards = cards[row_index * 3:(row_index + 1) * 3]

        for col, (icon, title, body) in zip(cols, row_cards):
            with col:
                st.markdown(
                    analysis_card(icon, title, body),
                    unsafe_allow_html=True,
                )

    # Catatan bawah
    note_text = (
        "Hasil SA, Tabu Search, dan VNS bergantung pada seed acak. "
        "Gunakan Multi-run >1 untuk menilai konsistensi hasil."
    )

    if multi:
        stats = {
            name: {
                "mean": float(np.mean([item["dist"] for item in values])),
                "std": float(np.std([item["dist"] for item in values])),
            }
            for name, values in multi.items()
            if values
        }

        if stats:
            min_std = min(value["std"] for value in stats.values())

            consistent_names = [
                name for name, value in stats.items()
                if abs(value["std"] - min_std) <= eps
            ]

            multi_text = "; ".join(
                f"{html.escape(name)}: rata-rata {value['mean']:.2f}, "
                f"SD {value['std']:.2f}"
                for name, value in stats.items()
            )

            note_text = (
                f"Multi-run ({int(runs_result)} run): {multi_text}. "
                f"Paling konsisten berdasarkan standar deviasi: "
                f"{', '.join(html.escape(x) for x in consistent_names)}. "
                + note_text
            )

    st.markdown(
        '<div class="analysis-note">'
        '<div class="analysis-icon">ℹ</div>'
        '<div><div class="analysis-note-title">Catatan</div>'
        f'<div>{note_text}</div></div></div>',
        unsafe_allow_html=True,
    )


# ---------- Tabu Explorer ----------

if n <= 12:
    with T["🔍 Tabu Explorer"]:
        ts_log = ts["log"]

        if ts_log:
            iteration = st.slider(
                "Iterasi",
                1,
                len(ts_log),
                1,
            )

            row = ts_log[iteration - 1]

            st.write(
                f"**Current:** "
                f"{' → '.join(map(str, row.get('Current_Before', [])))} "
                f"| **Tabu size:** {ts_size_result}"
            )

            st.dataframe(
                pd.DataFrame(row.get("Candidates", [])),
                width="stretch",
                hide_index=True,
            )

            st.write(
                f"**Move dipilih:** {row.get('Selected_Move', '-')}"
                f" → rute baru "
                f"{' → '.join(map(str, row.get('Current_After', [])))} "
                f"(jarak {row.get('Current_Distance', '-')})"
            )

            st.write(
                f"**Tabu list:** {row.get('Tabu_List', '-')}"
                f" | keluar dari list: {row.get('Left', '-')}"
            )


# ---------- Multi-run ----------

if multi:
    with T["🎲 Multi-run"]:
        summary = {}

        for name, values in multi.items():
            distances = [item["dist"] for item in values]
            runtimes = [item["time"] for item in values]

            summary[name] = {
                "Best": min(distances),
                "Worst": max(distances),
                "Mean": np.mean(distances),
                "Std Dev": np.std(distances),
                "Avg Runtime (s)": np.mean(runtimes),
            }

        multi_df = pd.DataFrame(summary).T.round(4)

        st.subheader("Ringkasan statistik")
        st.caption("Ringkasan ini merangkum seluruh run; tabel berikutnya memperlihatkan Run 1, Run 2, dan seterusnya secara individual.")
        st.dataframe(multi_df, width="stretch")

        detail_rows = []
        for method_name, values in multi.items():
            for run_idx, item in enumerate(values, start=1):
                detail_rows.append({
                    "Run": run_idx,
                    "Metode": method_name,
                    "Seed": int(seed) + run_idx - 1,
                    "Jarak akhir": float(item.get("dist", np.nan)),
                    "Improvement dari rute awal (%)": (100 * (d0 - float(item.get("dist", d0))) / d0) if d0 else 0.0,
                    "Iterasi aktual": item.get("iters", item.get("iterations", "—")),
                    "Runtime (detik)": float(item.get("time", 0.0)),
                    "Evaluasi kandidat": item.get("evaluations", "—"),
                })
        detail_df = pd.DataFrame(detail_rows)
        st.subheader("Hasil setiap run")
        st.dataframe(detail_df.round(4), width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Unduh detail setiap run (CSV)",
            detail_df.to_csv(index=False).encode("utf-8-sig"),
            "multirun_detail.csv",
            "text/csv",
        )

        st.download_button(
            "⬇️ Export multi-run",
            multi_df.to_csv().encode(),
            "multirun.csv",
            "text/csv",
        )
