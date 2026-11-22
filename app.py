"""
AgroScan - Dashboard interactivo con navegación lateral (Streamlit)
Tema: Green Fresh (inspirado en dashboard agrícola moderno).

Ejecutar:  python -m streamlit run app.py
Datos:     carpeta data/ con los CSV exportados de PostgreSQL (db_detector_maiz).
Opcional:  logo.png junto a app.py para usar tu propio logotipo.
"""
import base64
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AgroScan · Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== CONFIGURACIÓN =====
NOMBRE = "AgroScan"
SUBTITULO = "Detector de plagas del maíz"
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
LOGO = BASE / "logo.png"

UMBRAL = 80
META_EXACTITUD = 90
COLUMNAS_GRAFICAS = 2
ALTO = 360

# (clave, icono, etiqueta, título de página)
PAGINAS = [
    ("inicio",     "◫", "Inicio",                    "Inicio"),
    ("r1",         "◈", "R1 · Exactitud",            "R1 · Exactitud por clase y mes"),
    ("r2",         "◐", "R2 · Incidencia",           "R2 · Incidencia por parcela"),
    ("r3",         "⚠", "R3 · Confianza baja",       "R3 · Escaneos con confianza baja"),
    ("r4",         "⊞", "R4 · Matriz de confusión",  "R4 · Matriz de confusión"),
    ("r5",         "↗", "R5 · Evolución",            "R5 · Evolución temporal"),
    ("r6",         "⚙", "R6 · Algoritmos",           "R6 · Algoritmos y versiones"),
    ("decisiones", "◉", "Toma de decisiones",        "Toma de decisiones"),
]

# ===== Paleta verde / blanco =====
COLOR_BG          = "#eef2ec"
COLOR_SIDEBAR     = "#1c3c2c"
COLOR_SIDEBAR_DK  = "#122a1e"
COLOR_CARD        = "#ffffff"
COLOR_LIME        = "#c4f04a"
COLOR_LIME_SOFT   = "#e4f5b0"
COLOR_GREEN       = "#4a8b5c"
COLOR_GREEN_DEEP  = "#1a3d2e"
COLOR_GREEN_SOFT  = "#7bc99a"
COLOR_WARM        = "#e8a44d"
COLOR_TEXT        = "#1a2e1f"
COLOR_TEXTO       = COLOR_TEXT
COLOR_MUTED       = "#7a8a7f"
COLOR_BORDER      = "#e3ead9"

COLOR_META        = COLOR_WARM
COLOR_BARRA       = COLOR_GREEN_DEEP
COLOR_LINEA       = COLOR_GREEN

PALETA = ["#4a8b5c", "#c4f04a", "#1a3d2e", "#7bc99a",
          "#8ab890", "#d4e89c", "#2c5e3d", "#e8a44d"]

MIN_VERIF = 5


# ===== DATOS =====
def _csv(nombre):
    return pd.read_csv(DATA_DIR / nombre, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def cargar_datos():
    fact = _csv("fact_escaneos.csv")
    tiempo = _csv("dim_tiempo.csv")
    clase = _csv("dim_clase.csv")
    parcela = _csv("dim_parcela.csv")
    modelo = _csv("dim_modelo.csv")

    tiempo["fecha"] = pd.to_datetime(tiempo["fecha"], errors="coerce")
    tiempo["periodo"] = tiempo["fecha"].dt.strftime("%Y-%m")
    tiempo["dia"] = tiempo["fecha"].dt.strftime("%Y-%m-%d")
    tiempo["anio"] = tiempo["anio"].astype(str)
    tiempo["trimestre"] = "T" + tiempo["trimestre"].astype(str)
    tiempo["anio_trim"] = tiempo["anio"] + "-" + tiempo["trimestre"]

    ver = modelo["version_modelo"].astype(str)
    alg = modelo["algoritmo"].astype(str)
    modelo["modelo"] = np.where(ver.str.lower() == "desconocido", alg, alg + " " + ver)
    dup = modelo["modelo"].duplicated(keep=False)
    modelo.loc[dup, "modelo"] = modelo.loc[dup, "modelo"] + " (id " + modelo.loc[dup, "id_modelo"].astype(str) + ")"

    c_real = clase.rename(columns={"id_clase": "id_clase_real", "nombre_clase": "clase_real", "categoria": "categoria_real"})
    c_diag = clase.rename(columns={"id_clase": "id_diagnostico", "nombre_clase": "clase_diag", "categoria": "categoria_diag"})

    df = (
        fact.merge(tiempo[["id_tiempo", "fecha", "anio", "mes", "nombre_mes", "trimestre", "anio_trim", "periodo", "dia"]],
                   on="id_tiempo", how="left")
        .merge(c_real, on="id_clase_real", how="left")
        .merge(c_diag, on="id_diagnostico", how="left")
        .merge(parcela, on="id_parcela", how="left")
        .merge(modelo[["id_modelo", "algoritmo", "modelo"]], on="id_modelo", how="left")
        .rename(columns={"nombre_parcela": "parcela"})
    )

    for col in ["clase_real", "categoria_real", "clase_diag", "categoria_diag", "parcela", "algoritmo", "modelo"]:
        df[col] = df[col].fillna("desconocido").astype(str)

    df["confianza_pct"] = pd.to_numeric(df["confianza_pct"], errors="coerce")
    df["acierto"] = pd.to_numeric(df["acierto"], errors="coerce")
    df["cantidad_escaneos"] = pd.to_numeric(df["cantidad_escaneos"], errors="coerce").fillna(1)
    df["conf_baja"] = (df["confianza_pct"] < UMBRAL).astype(float).where(df["confianza_pct"].notna())
    return df


def resumen(df, por):
    n = df["cantidad_escaneos"]
    d = df.assign(
        _ok=df["acierto"] * n,
        _n=n.where(df["acierto"].notna(), 0),
        _b=df["conf_baja"] * n,
        _nb=n.where(df["conf_baja"].notna(), 0),
    )
    g = d.groupby(por, as_index=False).agg(
        escaneos=("cantidad_escaneos", "sum"),
        _ok=("_ok", "sum"),
        _n=("_n", "sum"),
        _b=("_b", "sum"),
        _nb=("_nb", "sum"),
        confianza=("confianza_pct", "mean"),
    )
    g["acierto_pct"] = g["_ok"] / g["_n"].replace(0, np.nan) * 100
    g["baja_pct"] = g["_b"] / g["_nb"].replace(0, np.nan) * 100
    g = g.rename(columns={"_n": "verificados", "_b": "n_baja"})
    return g.drop(columns=["_ok", "_nb"])


def precision(df):
    v = df[df["acierto"].notna()]
    n = v["cantidad_escaneos"].sum()
    if n == 0:
        return np.nan
    return (v["acierto"] * v["cantidad_escaneos"]).sum() / n * 100


def pct_baja(df):
    v = df[df["conf_baja"].notna()]
    n = v["cantidad_escaneos"].sum()
    if n == 0:
        return np.nan
    return (v["conf_baja"] * v["cantidad_escaneos"]).sum() / n * 100


def sin_desc(df, col):
    return df[df[col].astype(str).str.lower() != "desconocido"]


def cat_es(df, valores, col="categoria_diag"):
    return df[col].astype(str).str.lower().isin(valores)


def es_plaga(df, col="categoria_diag"):
    return cat_es(df, ["plaga"], col)


def es_incidencia(df, col="categoria_diag"):
    return cat_es(df, ["plaga", "enfermedad"], col)


def fmt_int(x):
    return f"{x:,.0f}"


def fmt_pct(x):
    return "s/d" if pd.isna(x) else f"{x:.1f}%"


def _mejor_peor(r, col="acierto_pct"):
    x = r.dropna(subset=[col])
    y = x[x["verificados"] >= MIN_VERIF]
    if y.empty:
        y = x
    if y.empty:
        return None, None
    return y.loc[y[col].idxmax()], y.loc[y[col].idxmin()]


def matriz_confusion(df):
    d = sin_desc(sin_desc(df, "clase_real"), "clase_diag")
    if d.empty:
        return None, None, None
    cm = d.pivot_table(index="clase_real", columns="clase_diag", values="cantidad_escaneos",
                       aggfunc="sum", fill_value=0)
    clases = sorted(set(cm.index) | set(cm.columns))
    cm = cm.reindex(index=clases, columns=clases, fill_value=0)
    tp = np.diag(cm.values).astype(float)
    pred = cm.sum(axis=0).values.astype(float)
    real = cm.sum(axis=1).values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        met = pd.DataFrame({
            "clase": clases,
            "precision": np.where(pred > 0, tp / pred * 100, np.nan),
            "sensibilidad": np.where(real > 0, tp / real * 100, np.nan),
            "real": real,
        })
    pares = d[d["clase_real"] != d["clase_diag"]].groupby(["clase_real", "clase_diag"], as_index=False)["cantidad_escaneos"].sum()
    pares = pares.sort_values("cantidad_escaneos", ascending=False)
    return cm, met, pares


def calcular_hallazgos(df):
    hall, recs = [], []
    total = df["cantidad_escaneos"].sum()
    if total == 0:
        return hall, recs

    prec = precision(df)
    if pd.notna(prec):
        cumple = prec >= META_EXACTITUD
        hall.append(("Exactitud del modelo",
                     f"Exactitud global de <b>{prec:.1f}%</b> frente a la meta de {META_EXACTITUD}%: "
                     + ("<b>cumple</b> la meta." if cumple else f"faltan <b>{META_EXACTITUD - prec:.1f}</b> puntos.")))
        recs.append("Mantener el modelo actual y monitorear su exactitud." if cumple else
                    f"Reentrenar el modelo: la exactitud ({prec:.1f}%) está por debajo de la meta de {META_EXACTITUD}%.")

    rm = resumen(sin_desc(df, "modelo"), "modelo")
    mejor, peor = _mejor_peor(rm)
    if mejor is not None:
        m = escape(str(mejor["modelo"]))
        hall.append(("Mejor versión del modelo", f"<b>{m}</b> alcanza {mejor['acierto_pct']:.1f}% de exactitud ({mejor['verificados']:.0f} escaneos verificados)."))
        recs.append(f"Priorizar la versión <b>{m}</b> en producción.")
        if peor is not None and peor["modelo"] != mejor["modelo"]:
            p = escape(str(peor["modelo"]))
            hall.append(("Versión con menor exactitud", f"<b>{p}</b> obtiene {peor['acierto_pct']:.1f}% ({peor['verificados']:.0f} escaneos verificados)."))
            recs.append(f"Revisar o reentrenar <b>{p}</b>, o retirarla si no mejora.")

    dp = sin_desc(df, "parcela")
    inc = dp[es_incidencia(dp)]
    if not inc.empty:
        pp = inc.groupby("parcela")["cantidad_escaneos"].sum().sort_values(ascending=False)
        top, n = escape(str(pp.index[0])), pp.iloc[0]
        hall.append(("Parcela más afectada", f"<b>{top}</b> concentra {fmt_int(n)} escaneos con plaga o enfermedad ({n / pp.sum() * 100:.0f}% del total de incidencias)."))
        recs.append(f"Priorizar el monitoreo y el control en <b>{top}</b>.")
        pm = inc.groupby("periodo")["cantidad_escaneos"].sum()
        if not pm.empty:
            hall.append(("Mes con mayor presión", f"<b>{escape(str(pm.idxmax()))}</b> fue el mes con más incidencias ({fmt_int(pm.max())})."))

    _, met, pares = matriz_confusion(df)
    if pares is not None and not pares.empty:
        r = pares.iloc[0]
        a, b = escape(str(r["clase_real"])), escape(str(r["clase_diag"]))
        hall.append(("Confusión más frecuente", f"El modelo diagnostica <b>{b}</b> cuando la clase real es <b>{a}</b> ({fmt_int(r['cantidad_escaneos'])} escaneos)."))
        recs.append(f"Agregar imágenes de <b>{a}</b> y <b>{b}</b> al banco de entrenamiento para que el modelo las distinga mejor.")

    baja = pct_baja(df)
    if pd.notna(baja):
        rb = resumen(sin_desc(df, "modelo"), "modelo").dropna(subset=["baja_pct"])
        txt = f"<b>{baja:.1f}%</b> de los escaneos tiene confianza menor a {UMBRAL}%."
        if not rb.empty:
            w = rb.loc[rb["baja_pct"].idxmax()]
            txt += f" La versión con más casos es <b>{escape(str(w['modelo']))}</b> ({w['baja_pct']:.1f}%)."
        hall.append(("Confianza baja", txt))
        recs.append(f"Revisar manualmente los escaneos con confianza menor a {UMBRAL}%.")

    sin_verif = df.loc[df["acierto"].isna(), "cantidad_escaneos"].sum() / total * 100
    if sin_verif > 0:
        hall.append(("Calidad de datos", f"<b>{sin_verif:.0f}%</b> de los escaneos no tiene la clase real del experto, así que no entra en la exactitud."))
        recs.append("Completar la verificación del experto para tener métricas más confiables.")
    return hall, recs


def tabla_decision(df):
    prec = precision(df)
    baja = pct_baja(df)
    total = df["cantidad_escaneos"].sum()
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()
    sin_verif = df.loc[df["acierto"].isna(), "cantidad_escaneos"].sum() / total * 100 if total else np.nan
    filas = [
        ("Exactitud global", fmt_pct(prec), f"≥ {META_EXACTITUD}%",
         "—" if pd.isna(prec) else ("✓ Cumple la meta" if prec >= META_EXACTITUD else "✗ Por debajo · reentrenar")),
        (f"Escaneos con confianza < {UMBRAL}%", fmt_pct(baja), "Lo más bajo posible", "→ Revisar manualmente"),
        ("Incidencia (plaga + enfermedad)", fmt_pct(inc / total * 100 if total else np.nan), "Lo más bajo posible", "→ Priorizar parcelas afectadas"),
        ("Escaneos sin verificar", fmt_pct(sin_verif), "0%", "→ Completar verificación" if sin_verif > 0 else "✓ Completo"),
    ]
    return pd.DataFrame(filas, columns=["Indicador", "Valor actual", "Meta", "Acción sugerida"])


# ===== CSS =====
CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.stApp {
    background: @@BG@@;
    min-height: 100vh;
}
.block-container { padding: 1rem 1.6rem 1.6rem 1.6rem; max-width: 100%; }
[data-testid="stMain"] :is(h1, h2, h3, h4, h5, p, label, li) { color: @@TEXT@@; }

/* ---------- TOPBAR ---------- */
.topbar {
    background: #ffffff;
    border-radius: 18px;
    padding: 0.75rem 1.1rem;
    margin-bottom: 1.3rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 1px 3px rgba(26,61,46,0.04), 0 4px 12px rgba(26,61,46,0.06);
    border: 1px solid #e8ede3;
}
.topbar-search {
    flex: 1 1 auto;
    max-width: 420px;
    background: #f5f7f0;
    border: 1px solid #e8ede3;
    border-radius: 10px;
    padding: 0.6rem 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    color: @@MUTED@@;
    font-size: 0.88rem;
}
.topbar-search .search-icon {
    color: @@GREEN@@;
    font-size: 1.05rem;
    line-height: 1;
}
.topbar-search .search-text {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.topbar-spacer { flex: 1; }
.topbar-icons {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.tb-btn {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #f5f7f0;
    color: @@GREEN@@;
    font-size: 1.05rem;
    transition: all 0.15s ease;
    cursor: pointer;
}
.tb-btn:hover { background: @@LIME_SOFT@@; color: @@GREEN_DEEP@@; }
.tb-divider {
    width: 1px;
    height: 28px;
    background: #e8ede3;
    margin: 0 0.5rem;
}
.tb-user {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding-left: 0.4rem;
}
.tb-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, @@LIME@@, @@GREEN_SOFT@@);
    color: @@GREEN_DEEP@@;
    font-weight: 800;
    font-size: 1rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(196,240,74,0.4);
}
.tb-user-text {
    display: flex;
    flex-direction: column;
    line-height: 1.2;
}
.tb-user-name {
    color: @@GREEN_DEEP@@;
    font-weight: 700;
    font-size: 0.88rem;
}
.tb-user-role {
    color: @@MUTED@@;
    font-size: 0.72rem;
    margin-top: 0.1rem;
}

@media (max-width: 900px) {
    .topbar-search { display: none; }
    .tb-user-text { display: none; }
}

/* ---------- SIDEBAR ---------- */
.st-key-menu {
    background: linear-gradient(180deg, @@SIDEBAR@@ 0%, @@SIDEBAR_DK@@ 100%);
    border-radius: 22px;
    padding: 1.3rem 1rem 1.3rem 1rem;
    box-shadow: 0 10px 30px rgba(26,61,46,0.28);
    border: 1px solid rgba(196,240,74,0.08);
}
.st-key-menu [data-testid="stVerticalBlock"] { gap: 0.2rem; }

.st-key-menu .stButton button {
    width: 100%; min-height: 2.55rem; height: auto;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.55rem 0.85rem !important;
    transition: all 0.18s ease;
}
.st-key-menu .stButton button p {
    font-size: 0.86rem;
    text-align: left;
    color: #b8d4c2 !important;
    font-weight: 500;
}
.st-key-menu button[kind="secondary"]:hover,
.st-key-menu [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(196,240,74,0.08) !important;
}
.st-key-menu button[kind="secondary"]:hover p,
.st-key-menu [data-testid="stBaseButton-secondary"]:hover p {
    color: #ffffff !important;
}
.st-key-menu button[kind="primary"],
.st-key-menu [data-testid="stBaseButton-primary"] {
    background: @@LIME@@ !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(196,240,74,0.3) !important;
}
.st-key-menu button[kind="primary"] p,
.st-key-menu [data-testid="stBaseButton-primary"] p {
    color: @@GREEN_DEEP@@ !important;
    font-weight: 800;
}

.st-key-menu :is(label, [data-testid="stWidgetLabel"] p) {
    color: #8fa89a !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}
.st-key-menu [data-baseweb="select"] > div {
    background: rgba(0,0,0,0.22) !important;
    border-color: rgba(196,240,74,0.18) !important;
    color: #e0edde !important;
    border-radius: 8px !important;
}
.st-key-menu [data-baseweb="select"] span { color: #e0edde !important; }
.st-key-menu [data-baseweb="tag"] {
    background: rgba(196,240,74,0.2) !important;
    color: #e0edde !important;
}

.brand {
    padding: 0.3rem 0.3rem 0.9rem;
    border-bottom: 1px solid rgba(196,240,74,0.12);
    margin-bottom: 0.9rem;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.brand-logo {
    width: 42px;
    height: 42px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(196,240,74,0.15);
    border-radius: 11px;
    padding: 6px;
}
.brand-logo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}
.brand-text {
    flex: 1;
    min-width: 0;
    line-height: 1.15;
}
.brand-n {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.brand-s {
    color: #8fa89a;
    font-size: 0.68rem;
    margin-top: 0.15rem;
}

.filtros-t {
    color: #8fa89a;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    border-top: 1px solid rgba(196,240,74,0.12);
    padding-top: 0.85rem;
    margin-top: 0.7rem;
    margin-bottom: 0.5rem;
}

.mini-stat {
    background: rgba(196,240,74,0.1);
    border: 1px solid rgba(196,240,74,0.22);
    border-radius: 12px;
    padding: 0.85rem 0.9rem;
    margin-top: 0.7rem;
}
.mini-stat-title {
    color: #8fa89a;
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.mini-stat-value {
    color: @@LIME@@;
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.mini-stat-bar {
    height: 5px;
    background: rgba(0,0,0,0.28);
    border-radius: 3px;
    overflow: hidden;
}
.mini-stat-fill {
    height: 100%;
    background: @@LIME@@;
    border-radius: 3px;
}

/* ---------- PAGE TITLE ---------- */
.page-title-block {
    margin: 0 0 1.4rem 0;
}
.page-crumb {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: @@MUTED@@;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.page-title {
    font-size: 1.65rem;
    color: @@GREEN_DEEP@@;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.01em;
}

/* ---------- KPI ---------- */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e8ede3;
    border-radius: 18px;
    padding: 1.2rem 1rem 1rem;
    text-align: center;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(26,61,46,0.04), 0 4px 12px rgba(26,61,46,0.06);
    position: relative;
    overflow: hidden;
}
.kpi-gauge-wrap {
    position: relative;
    width: 96px;
    height: 96px;
    margin: 0 auto 0.6rem;
}
.kpi-gauge { width: 100%; height: 100%; display: block; }
.kpi-number {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.15rem;
    font-weight: 800;
    color: @@GREEN_DEEP@@;
    white-space: nowrap;
    letter-spacing: -0.02em;
}
.kpi-value-big {
    font-size: 1.55rem;
    font-weight: 800;
    color: @@GREEN_DEEP@@;
    line-height: 1.15;
    margin: 0.9rem 0 0.55rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: -0.02em;
}
.kpi-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: @@MUTED@@;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #9caa9e;
}

/* ---------- INFO CARD ---------- */
.info-card {
    background: #ffffff;
    border: 1px solid #e8ede3;
    border-radius: 16px;
    padding: 1.15rem 1.3rem;
    height: 100%;
    box-shadow: 0 1px 3px rgba(26,61,46,0.03), 0 4px 10px rgba(26,61,46,0.04);
}
.info-card .lbl {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: @@GREEN@@;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.info-card .txt {
    color: @@TEXT@@;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ---------- SECTION ---------- */
.section-h {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.8rem 0 1.1rem 0;
}
.section-n {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: @@LIME@@;
    color: @@GREEN_DEEP@@;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.82rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(196,240,74,0.35);
}
.section-t {
    font-size: 1.1rem;
    color: @@GREEN_DEEP@@;
    font-weight: 800;
    letter-spacing: -0.005em;
}

/* ---------- HALLAZGO ---------- */
.hallazgo {
    background: #f7fbe9;
    border-left: 3px solid @@LIME@@;
    color: @@TEXT@@;
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.15rem;
    margin: 0.9rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.hallazgo b { color: @@GREEN_DEEP@@; font-weight: 800; }

/* ---------- CARD ---------- */
.card {
    background: #ffffff;
    border: 1px solid #e8ede3;
    border-left: 3px solid @@GREEN@@;
    border-radius: 12px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 0.7rem;
    min-height: 6rem;
    box-shadow: 0 1px 3px rgba(26,61,46,0.03);
}
.card-t {
    color: @@GREEN_DEEP@@;
    font-weight: 800;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
.card-x {
    color: @@TEXT@@;
    font-size: 0.87rem;
    line-height: 1.55;
}
.card-x b { color: @@GREEN_DEEP@@; font-weight: 800; }

/* ---------- RECO ---------- */
.reco-box {
    background: #f0f7ee;
    border-left: 3px solid @@GREEN@@;
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.15rem;
    margin-bottom: 0.6rem;
    color: @@TEXT@@;
    font-size: 0.9rem;
    line-height: 1.6;
}
.reco-box b { color: @@GREEN_DEEP@@; font-weight: 800; }
.reco-box b.principal {
    color: @@GREEN_DEEP@@;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.78rem;
    display: block;
    margin-bottom: 0.25rem;
}

/* ---------- PIE ---------- */
.pie {
    color: @@MUTED@@;
    font-size: 0.72rem;
    margin-top: 1.4rem;
    padding-top: 0.8rem;
    border-top: 1px solid #e8ede3;
    letter-spacing: 0.02em;
}

/* ---------- FORM ---------- */
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {
    font-size: 0.73rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: @@MUTED@@ !important;
    font-weight: 700;
}
[data-testid="stMain"] [data-testid="stRadio"] label p {
    color: @@TEXT@@ !important;
    font-size: 0.85rem !important;
    text-transform: none;
    letter-spacing: 0;
    font-weight: 500;
}

/* ---------- ST.TABLE ---------- */
[data-testid="stTable"] table {
    background: #ffffff;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e8ede3;
    box-shadow: 0 1px 3px rgba(26,61,46,0.03);
}
[data-testid="stTable"] thead tr th {
    background: #f5f8f0 !important;
    color: @@GREEN_DEEP@@ !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em;
    padding: 0.75rem 0.9rem !important;
    border-bottom: 1px solid #e8ede3 !important;
    text-align: left !important;
}
[data-testid="stTable"] tbody tr td {
    color: @@TEXT@@ !important;
    padding: 0.7rem 0.9rem !important;
    border-bottom: 1px solid #f0f3ec !important;
    font-size: 0.86rem !important;
    background: #ffffff !important;
}
[data-testid="stTable"] tbody tr:hover td { background: #fafcf6 !important; }

/* ---------- ALERT ---------- */
[data-testid="stAlert"] {
    background: #f5f8f0 !important;
    border-left: 3px solid @@GREEN@@ !important;
    color: @@TEXT@@ !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] p { color: @@TEXT@@ !important; }

/* ---------- CAPTION ---------- */
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: @@MUTED@@ !important;
    font-size: 0.75rem !important;
}
[data-testid="stMain"] .stMarkdown p { color: @@TEXT@@; }
</style>
"""
for token, valor in {
    "@@BG@@": COLOR_BG,
    "@@SIDEBAR@@": COLOR_SIDEBAR,
    "@@SIDEBAR_DK@@": COLOR_SIDEBAR_DK,
    "@@LIME@@": COLOR_LIME,
    "@@LIME_SOFT@@": COLOR_LIME_SOFT,
    "@@GREEN@@": COLOR_GREEN,
    "@@GREEN_DEEP@@": COLOR_GREEN_DEEP,
    "@@GREEN_SOFT@@": COLOR_GREEN_SOFT,
    "@@TEXT@@": COLOR_TEXT,
    "@@MUTED@@": COLOR_MUTED,
}.items():
    CSS = CSS.replace(token, valor)
st.markdown(CSS, unsafe_allow_html=True)

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="30" fill="#c4f04a"/>
<path d="M32 16 C 22 22, 18 32, 20 46 C 26 48, 34 44, 38 36 C 42 28, 42 20, 36 16 Z" fill="#1a3d2e"/>
<path d="M32 20 L 32 42 M 26 30 L 32 34 M 38 28 L 32 32" stroke="#c4f04a" stroke-width="2" fill="none" stroke-linecap="round"/>
</svg>"""


def logo_html():
    if LOGO.exists():
        datos, tipo = base64.b64encode(LOGO.read_bytes()).decode(), "image/png"
    else:
        datos, tipo = base64.b64encode(LOGO_SVG.encode()).decode(), "image/svg+xml"
    return f'<img src="data:{tipo};base64,{datos}" alt="logo">'


# ----- utilidades de dibujo -----
def boton(label, **kw):
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:
        return st.button(label, use_container_width=True, **kw)


def mostrar(fig, key):
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key,
                        config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key,
                        config={"displayModeBar": False})


def estilo(fig, titulo, x="", y=""):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left",
                   font=dict(size=14, color=COLOR_GREEN_DEEP,
                             family="Inter, Segoe UI, sans-serif", weight="bold")),
        height=ALTO, margin=dict(l=60, r=20, t=55, b=55),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(color=COLOR_TEXT, family="Inter, Segoe UI, sans-serif", size=11),
        legend=dict(title=None, font=dict(size=10, color=COLOR_TEXT),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=COLOR_TEXT, size=11),
                        bordercolor=COLOR_GREEN),
    )
    fig.update_xaxes(title_text=x, showgrid=False, automargin=True,
                     title_font=dict(size=11, color=COLOR_MUTED),
                     tickfont=dict(size=10, color=COLOR_MUTED),
                     linecolor="#e8ede3")
    fig.update_yaxes(title_text=y, gridcolor="#f0f3ec", griddash="dot",
                     zeroline=False, automargin=True,
                     title_font=dict(size=11, color=COLOR_MUTED),
                     tickfont=dict(size=10, color=COLOR_MUTED),
                     linecolor="#e8ede3")
    return fig


def rejilla(items):
    n = max(1, COLUMNAS_GRAFICAS)
    for i in range(0, len(items), n):
        cols = st.columns(n, gap="large")
        for col, (clave, fig) in zip(cols, items[i:i + n]):
            with col:
                if isinstance(fig, str):
                    st.info(fig)
                else:
                    mostrar(fig, clave)


def sin_datos(titulo):
    return f"{titulo}: sin datos con los filtros actuales."


def meta_h(fig, valor=META_EXACTITUD):
    fig.add_hline(y=valor, line_dash="dash", line_color=COLOR_META,
                  line_width=1.5,
                  annotation_text=f"Meta {valor}%",
                  annotation_position="top left",
                  annotation_font=dict(color=COLOR_META, size=10))
    return fig


def _auto_pct(valor):
    if isinstance(valor, str) and valor.strip().endswith('%'):
        try:
            return float(valor.strip().rstrip('%'))
        except ValueError:
            return None
    return None


def kpi(col, titulo, valor, sub="", color=COLOR_GREEN):
    pct = _auto_pct(valor)
    if pct is not None:
        pct_c = max(0.0, min(100.0, pct))
        dash = pct_c * 2.513
        inner = f"""
        <div class='kpi-gauge-wrap'>
            <svg class='kpi-gauge' viewBox='0 0 100 100'>
                <circle cx='50' cy='50' r='40' stroke='#f0f3ec' stroke-width='9' fill='none'/>
                <circle cx='50' cy='50' r='40' stroke='{color}' stroke-width='9' fill='none'
                        stroke-dasharray='{dash} 251' stroke-linecap='round'
                        transform='rotate(-90 50 50)'/>
            </svg>
            <div class='kpi-number'>{valor}</div>
        </div>
        """
    else:
        inner = f"<div class='kpi-value-big'>{valor}</div>"
    col.markdown(
        f"<div class='kpi-card'>{inner}"
        f"<div class='kpi-title'>{titulo}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def hallazgo(texto):
    st.markdown(f"<div class='hallazgo'><b>Hallazgo.</b> {texto}</div>", unsafe_allow_html=True)


def encabezado(clave, titulo):
    num = next((p[1] for p in PAGINAS if p[0] == clave), "00")
    st.markdown(
        f"<div class='page-title-block'>"
        f"<div class='page-crumb'>{NOMBRE} · Sección {num}</div>"
        f"<div class='page-title'>{escape(titulo)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def seccion(num, titulo):
    st.markdown(
        f"<div class='section-h'><div class='section-n'>{num}</div>"
        f"<div class='section-t'>{escape(titulo)}</div></div>",
        unsafe_allow_html=True,
    )


def nota_desc(df, d):
    n = df["cantidad_escaneos"].sum() - d["cantidad_escaneos"].sum()
    if n > 0:
        st.caption(f"{fmt_int(n)} escaneos con valor «desconocido» no se muestran en estas gráficas.")


def barra_h(r, x, y, titulo, fmt, xt, color=None):
    r = r.sort_values(x, ascending=True)
    fig = px.bar(r, x=x, y=y, orientation="h", text_auto=fmt, color=color,
                 color_discrete_sequence=PALETA if color else [COLOR_BARRA])
    fig.update_traces(textfont=dict(color=COLOR_TEXT, size=10), marker_line_width=0)
    return estilo(fig, titulo, x=xt)


# ===== TOPBAR =====
def topbar():
    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-search">
                <span class="search-icon">⌕</span>
                <span class="search-text">Buscar parcela, clase o modelo…</span>
            </div>
            <div class="topbar-spacer"></div>
            <div class="topbar-icons">
                <span class="tb-btn" title="Notificaciones">◔</span>
                <span class="tb-btn" title="Configuración">⚙</span>
                <span class="tb-btn" title="Mensajes">✉</span>
                <span class="tb-divider"></span>
                <div class="tb-user">
                    <span class="tb-avatar">A</span>
                    <div class="tb-user-text">
                        <span class="tb-user-name">AgroScan</span>
                        <span class="tb-user-role">Investigador · ITT</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===== PÁGINAS =====
def pagina_inicio(df):
    total = df["cantidad_escaneos"].sum()
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Objetivo</div>"
            "<div class='txt'>Analizar a lo largo del tiempo los diagnósticos del detector inteligente del gusano "
            "cogollero y la roya del maíz del Instituto Tecnológico de Tizimín, para decidir en qué parcelas y "
            "periodos intervenir y cuándo conviene reentrenar el modelo.</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Usuarios</div>"
            "<div class='txt'>Docente y estudiantes de Agronomía, y el equipo de desarrollo del proyecto.</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Cómo usarlo</div>"
            "<div class='txt'>Elija una sección en el menú. Las secciones R1 a R6 responden los seis requerimientos "
            "del data mart. «Toma de decisiones» concentra los hallazgos y las acciones sugeridas.</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    k = st.columns(4)
    kpi(k[0], "Total de escaneos", fmt_int(total), f"{fmt_int(len(df))} registros", color=COLOR_GREEN_DEEP)
    kpi(k[1], "Exactitud del modelo", fmt_pct(precision(df)), f"meta {META_EXACTITUD}%", color=COLOR_GREEN)
    kpi(k[2], "Confianza promedio", fmt_pct(df["confianza_pct"].mean()), "reportada por el modelo", color=COLOR_LIME)
    kpi(k[3], "Plaga o enfermedad", fmt_int(inc), f"{inc / total * 100:.1f}% de los escaneos", color=COLOR_GREEN_SOFT)

    por_mes = resumen(df, "periodo")
    fig1 = px.line(por_mes, x="periodo", y="escaneos", markers=True,
                   color_discrete_sequence=[COLOR_GREEN_DEEP])
    fig1.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME,
                                                    line=dict(color=COLOR_GREEN_DEEP, width=1.5)))
    fig1.update_xaxes(type="category")
    por_cat = resumen(df, "categoria_diag").sort_values("escaneos", ascending=False)
    fig2 = px.bar(por_cat, x="categoria_diag", y="escaneos", text_auto=".0f",
                  color_discrete_sequence=PALETA)
    fig2.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    rejilla([
        ("i_mes", estilo(fig1, "Escaneos por mes", x="Mes", y="Escaneos")),
        ("i_cat", estilo(fig2, "Escaneos por categoría diagnosticada", y="Escaneos")),
    ])

    top_mes = por_mes.loc[por_mes["escaneos"].idxmax()]
    top_cat = por_cat.iloc[0]
    hallazgo(
        f"El mes con más actividad fue <b>{escape(str(top_mes['periodo']))}</b> ({fmt_int(top_mes['escaneos'])} escaneos) "
        f"y la categoría más diagnosticada es <b>{escape(str(top_cat['categoria_diag']))}</b> "
        f"({top_cat['escaneos'] / total * 100:.0f}% del total)."
    )


def pagina_r1(df):
    d = sin_desc(df, "clase_real")
    if d["acierto"].notna().sum() == 0:
        st.info(sin_datos("R1"))
        return
    prec = precision(d)
    rc = resumen(d, "clase_real")
    mejor, peor = _mejor_peor(rc)
    verif = d.loc[d["acierto"].notna(), "cantidad_escaneos"].sum()

    k = st.columns(4)
    kpi(k[0], "Exactitud global", fmt_pct(prec),
        "✓ cumple la meta" if prec >= META_EXACTITUD else f"faltan {META_EXACTITUD - prec:.1f} pts",
        color=COLOR_GREEN)
    kpi(k[1], "Clase con mayor exactitud", escape(str(mejor["clase_real"])) if mejor is not None else "s/d",
        fmt_pct(mejor["acierto_pct"]) if mejor is not None else "", color=COLOR_LIME)
    kpi(k[2], "Clase con menor exactitud", escape(str(peor["clase_real"])) if peor is not None else "s/d",
        fmt_pct(peor["acierto_pct"]) if peor is not None else "", color=COLOR_GREEN_SOFT)
    kpi(k[3], "Escaneos verificados", fmt_int(verif), "con clase real del experto", color=COLOR_GREEN_DEEP)

    items = []
    x = rc.dropna(subset=["acierto_pct"]).sort_values("acierto_pct", ascending=False)
    fig = px.bar(x, x="clase_real", y="acierto_pct", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r1_clase", meta_h(estilo(fig, "Exactitud por clase (%)", y="Exactitud %"))))

    x = resumen(d, "periodo").dropna(subset=["acierto_pct"])
    if x.empty:
        items.append(("r1_mes", sin_datos("Exactitud por mes")))
    else:
        fig = px.line(x, x="periodo", y="acierto_pct", markers=True, color_discrete_sequence=[COLOR_GREEN_DEEP])
        fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME,
                                                      line=dict(color=COLOR_GREEN_DEEP, width=1.5)))
        fig.update_xaxes(type="category")
        items.append(("r1_mes", meta_h(estilo(fig, "Exactitud por mes (%)", x="Mes", y="Exactitud %"))))

    x = resumen(d, "anio_trim").dropna(subset=["acierto_pct"])
    if x.empty:
        items.append(("r1_trim", sin_datos("Exactitud por trimestre")))
    else:
        fig = px.bar(x, x="anio_trim", y="acierto_pct", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
        fig.update_xaxes(type="category")
        items.append(("r1_trim", meta_h(estilo(fig, "Exactitud por trimestre (%)", y="Exactitud %"))))

    x = resumen(sin_desc(d, "modelo"), "modelo").dropna(subset=["acierto_pct"])
    if x.empty:
        items.append(("r1_mod", sin_datos("Exactitud por versión del modelo")))
    else:
        fig = barra_h(x, "acierto_pct", "modelo", "Exactitud por versión del modelo (%)", ".1f", "Exactitud %")
        fig.add_vline(x=META_EXACTITUD, line_dash="dash", line_color=COLOR_META, line_width=1.5)
        items.append(("r1_mod", fig))
    rejilla(items)
    nota_desc(df, d)

    texto = (f"La exactitud global es <b>{prec:.1f}%</b>: " +
             ("cumple la meta de " if prec >= META_EXACTITUD else "no alcanza la meta de ") + f"{META_EXACTITUD}%.")
    if peor is not None and mejor is not None and peor["clase_real"] != mejor["clase_real"]:
        texto += (f" Va de {peor['acierto_pct']:.1f}% en <b>{escape(str(peor['clase_real']))}</b> "
                  f"a {mejor['acierto_pct']:.1f}% en <b>{escape(str(mejor['clase_real']))}</b>.")
    hallazgo(texto)


def pagina_r2(df):
    d = sin_desc(df, "parcela")
    if d.empty:
        st.info(sin_datos("R2"))
        return
    total = d["cantidad_escaneos"].sum()
    plaga = d.loc[es_plaga(d), "cantidad_escaneos"].sum()
    enf = d.loc[cat_es(d, ["enfermedad"]), "cantidad_escaneos"].sum()
    inc = d.loc[es_incidencia(d), "cantidad_escaneos"].sum()

    g = d.assign(_inc=np.where(es_incidencia(d), d["cantidad_escaneos"], 0)).groupby("parcela", as_index=False).agg(
        escaneos=("cantidad_escaneos", "sum"), incidencias=("_inc", "sum"))
    g["tasa"] = g["incidencias"] / g["escaneos"] * 100
    top = g.loc[g["incidencias"].idxmax()]

    k = st.columns(4)
    kpi(k[0], "Plaga (cogollero)", fmt_int(plaga), f"{plaga / total * 100:.1f}% de los escaneos", color=COLOR_GREEN)
    kpi(k[1], "Enfermedad (roya)", fmt_int(enf), f"{enf / total * 100:.1f}% de los escaneos", color=COLOR_LIME)
    kpi(k[2], "Incidencia total", f"{inc / total * 100:.1f}%", f"{fmt_int(inc)} escaneos", color=COLOR_GREEN_SOFT)
    kpi(k[3], "Parcela más afectada", escape(str(top["parcela"])), f"{fmt_int(top['incidencias'])} incidencias", color=COLOR_GREEN_DEEP)

    items = []
    rcat = resumen(d, ["parcela", "categoria_diag"])
    fig = px.bar(rcat, x="parcela", y="escaneos", color="categoria_diag", barmode="stack",
                 color_discrete_sequence=PALETA)
    fig.update_layout(barnorm="percent")
    fig.update_traces(marker_line_width=0)
    items.append(("r2_stack", estilo(fig, "Diagnóstico por parcela (% de escaneos)", y="% de escaneos")))

    fig = px.bar(g.sort_values("tasa", ascending=False), x="parcela", y="tasa", text_auto=".1f",
                 color_discrete_sequence=[COLOR_BARRA])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r2_tasa", estilo(fig, "Incidencia por parcela (% de escaneos)", y="Incidencia %")))

    di = d[es_incidencia(d)]
    if di.empty:
        items.append(("r2_mes", sin_datos("Incidencia por mes")))
        items.append(("r2_heat", sin_datos("Incidencia por parcela y trimestre")))
    else:
        rm = resumen(di, ["periodo", "categoria_diag"])
        fig = px.line(rm, x="periodo", y="escaneos", color="categoria_diag", markers=True,
                      color_discrete_sequence=PALETA)
        fig.update_traces(line_width=2.5, marker=dict(size=6))
        fig.update_xaxes(type="category")
        items.append(("r2_mes", estilo(fig, "Plaga y enfermedad por mes", x="Mes", y="Escaneos")))
        mat = di.pivot_table(index="parcela", columns="anio_trim", values="cantidad_escaneos", aggfunc="sum", fill_value=0)
        fig = px.imshow(mat, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, "#f0f7e8"], [0.3, "#d9f58c"], [0.6, "#7bc99a"], [1, "#1a3d2e"]],
                        labels=dict(x="Trimestre", y="Parcela"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textfont=dict(color=COLOR_TEXT))
        items.append(("r2_heat", estilo(fig, "Incidencias por parcela y trimestre", x="Trimestre", y="Parcela")))
    rejilla(items)
    nota_desc(df, d)
    st.caption("Incidencia = escaneos diagnosticados como plaga o enfermedad. La zona de la parcela no está en las tablas exportadas, por eso se analiza por parcela.")

    hallazgo(f"<b>{escape(str(top['parcela']))}</b> es la parcela con más incidencias ({fmt_int(top['incidencias'])}; "
             f"{top['tasa']:.1f}% de sus escaneos). En total, {inc / total * 100:.1f}% de los escaneos resultó en plaga o enfermedad.")


def pagina_r3(df):
    d = df[df["conf_baja"].notna()]
    if d.empty:
        st.info(sin_datos("R3"))
        return
    baja = pct_baja(d)
    n_baja = (d["conf_baja"] * d["cantidad_escaneos"]).sum()
    rm = resumen(sin_desc(d, "modelo"), "modelo").dropna(subset=["baja_pct"])
    peor = rm.loc[rm["baja_pct"].idxmax()] if not rm.empty else None

    k = st.columns(4)
    kpi(k[0], "Umbral de confianza", f"{UMBRAL}%", "debajo = confianza baja", color=COLOR_GREEN)
    kpi(k[1], "Escaneos con confianza baja", fmt_int(n_baja), f"{baja:.1f}% de los escaneos", color=COLOR_LIME)
    kpi(k[2], "Versión con más casos", escape(str(peor["modelo"])) if peor is not None else "s/d",
        fmt_pct(peor["baja_pct"]) if peor is not None else "", color=COLOR_GREEN_SOFT)
    kpi(k[3], "Confianza promedio", fmt_pct(d["confianza_pct"].mean()), "de todos los escaneos", color=COLOR_GREEN_DEEP)

    items = []
    if rm.empty:
        items.append(("r3_mod", sin_datos("Confianza baja por versión")))
    else:
        items.append(("r3_mod", barra_h(rm, "baja_pct", "modelo", "Confianza baja por versión (%)", ".1f", "% con confianza baja")))
    ra = resumen(sin_desc(d, "algoritmo"), "algoritmo").dropna(subset=["baja_pct"])
    if ra.empty:
        items.append(("r3_alg", sin_datos("Confianza baja por algoritmo")))
    else:
        fig = px.bar(ra.sort_values("baja_pct", ascending=False), x="algoritmo", y="baja_pct", text_auto=".1f",
                     color_discrete_sequence=[COLOR_BARRA])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
        items.append(("r3_alg", estilo(fig, "Confianza baja por algoritmo (%)", y="% con confianza baja")))
    fig = px.histogram(d, x="confianza_pct", nbins=20, color_discrete_sequence=[COLOR_GREEN])
    fig.update_traces(marker_line_width=0)
    fig.add_vline(x=UMBRAL, line_dash="dash", line_color=COLOR_META,
                  annotation_text=f"Umbral {UMBRAL}%",
                  annotation_font=dict(color=COLOR_META))
    items.append(("r3_hist", estilo(fig, "Distribución de la confianza", x="Confianza %", y="Escaneos")))
    pm = resumen(d, "periodo").dropna(subset=["baja_pct"])
    fig = px.line(pm, x="periodo", y="baja_pct", markers=True, color_discrete_sequence=[COLOR_GREEN_DEEP])
    fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME,
                                                  line=dict(color=COLOR_GREEN_DEEP, width=1.5)))
    fig.update_xaxes(type="category")
    items.append(("r3_mes", estilo(fig, "Confianza baja por mes (%)", x="Mes", y="% con confianza baja")))
    rejilla(items)
    st.caption("La fuente de la imagen (prototipo, campo o banco) no está en las tablas exportadas, por eso se analiza por versión y algoritmo.")

    texto = f"<b>{baja:.1f}%</b> de los escaneos tiene confianza menor a {UMBRAL}%."
    if peor is not None:
        texto += f" La versión con más casos es <b>{escape(str(peor['modelo']))}</b> ({peor['baja_pct']:.1f}%): conviene revisar sus diagnósticos manualmente."
    hallazgo(texto)


def pagina_r4(df):
    cm, met, pares = matriz_confusion(df)
    if cm is None:
        st.info(sin_datos("R4"))
        return
    total = cm.values.sum()
    ok = np.trace(cm.values)
    m = met.dropna(subset=["sensibilidad"])
    m = m[m["real"] >= 1]
    mejor = m.loc[m["sensibilidad"].idxmax()] if not m.empty else None
    peor = m.loc[m["sensibilidad"].idxmin()] if not m.empty else None

    k = st.columns(4)
    kpi(k[0], "Escaneos comparados", fmt_int(total), "clase real y diagnóstico", color=COLOR_GREEN)
    kpi(k[1], "Diagnósticos correctos", f"{ok / total * 100:.1f}%", f"{fmt_int(ok)} de {fmt_int(total)}", color=COLOR_LIME)
    kpi(k[2], "Mejor sensibilidad", escape(str(mejor["clase"])) if mejor is not None else "s/d",
        fmt_pct(mejor["sensibilidad"]) if mejor is not None else "", color=COLOR_GREEN_SOFT)
    kpi(k[3], "Peor sensibilidad", escape(str(peor["clase"])) if peor is not None else "s/d",
        fmt_pct(peor["sensibilidad"]) if peor is not None else "", color=COLOR_GREEN_DEEP)

    items = []
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0, "#f0f7e8"], [0.3, "#d9f58c"], [0.6, "#7bc99a"], [1, "#1a3d2e"]],
                    labels=dict(x="Diagnóstico del modelo", y="Clase real"))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont=dict(color=COLOR_TEXT))
    items.append(("r4_heat", estilo(fig, "Clase real vs. diagnóstico", x="Diagnóstico del modelo", y="Clase real")))

    d = sin_desc(sin_desc(df, "categoria_real"), "categoria_diag")
    if d.empty:
        items.append(("r4_cat", sin_datos("Matriz por categoría")))
    else:
        mc = d.pivot_table(index="categoria_real", columns="categoria_diag", values="cantidad_escaneos",
                           aggfunc="sum", fill_value=0)
        fig = px.imshow(mc, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, "#f0f7e8"], [0.3, "#d9f58c"], [0.6, "#7bc99a"], [1, "#1a3d2e"]],
                        labels=dict(x="Categoría diagnosticada", y="Categoría real"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textfont=dict(color=COLOR_TEXT))
        items.append(("r4_cat", estilo(fig, "Por categoría de enfermedad", x="Categoría diagnosticada", y="Categoría real")))

    largo = met.melt(id_vars="clase", value_vars=["precision", "sensibilidad"], var_name="métrica", value_name="valor")
    largo["métrica"] = largo["métrica"].map({"precision": "Precisión", "sensibilidad": "Sensibilidad"})
    fig = px.bar(largo, x="clase", y="valor", color="métrica", barmode="group", text_auto=".0f",
                 color_discrete_sequence=[COLOR_GREEN_DEEP, COLOR_LIME])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r4_met", estilo(fig, "Precisión y sensibilidad por clase (%)", y="%")))

    if pares is None or pares.empty:
        items.append(("r4_par", "No hay confusiones entre clases con los filtros actuales."))
    else:
        p = pares.head(6).copy()
        p["par"] = p["clase_real"] + " → " + p["clase_diag"]
        items.append(("r4_par", barra_h(p, "cantidad_escaneos", "par", "Confusiones más frecuentes (real → diagnóstico)", ".0f", "Escaneos")))
    rejilla(items)

    texto = f"<b>{ok / total * 100:.1f}%</b> de los diagnósticos coincide con la clase real."
    if pares is not None and not pares.empty:
        r = pares.iloc[0]
        texto += (f" La confusión más frecuente es <b>{escape(str(r['clase_real']))}</b> clasificada como "
                  f"<b>{escape(str(r['clase_diag']))}</b> ({fmt_int(r['cantidad_escaneos'])} escaneos).")
    hallazgo(texto)


NIVELES = {"Año": "anio", "Trimestre": "anio_trim", "Mes": "periodo", "Día": "dia"}


def pagina_r5(df):
    nivel = st.radio("Nivel de detalle (año → trimestre → mes → día)", list(NIVELES),
                     index=2, horizontal=True, key="nivel_tiempo")
    col = NIVELES[nivel]
    por = resumen(df, col)
    di = df[es_incidencia(df)]
    pico = por.loc[por["escaneos"].idxmax()]
    pico_inc = di.groupby(col)["cantidad_escaneos"].sum() if not di.empty else None

    k = st.columns(4)
    kpi(k[0], "Escaneos", fmt_int(df["cantidad_escaneos"].sum()), f"nivel: {nivel.lower()}", color=COLOR_GREEN)
    kpi(k[1], f"{nivel} con más escaneos", escape(str(pico[col])), f"{fmt_int(pico['escaneos'])} escaneos", color=COLOR_LIME)
    kpi(k[2], "Confianza promedio", fmt_pct(df["confianza_pct"].mean()), "de todos los escaneos", color=COLOR_GREEN_SOFT)
    kpi(k[3], f"Pico de incidencia", escape(str(pico_inc.idxmax())) if pico_inc is not None else "s/d",
        f"{fmt_int(pico_inc.max())} casos" if pico_inc is not None else "", color=COLOR_GREEN_DEEP)

    items = []
    fig = px.line(por, x=col, y="escaneos", markers=True, color_discrete_sequence=[COLOR_GREEN_DEEP])
    fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME,
                                                  line=dict(color=COLOR_GREEN_DEEP, width=1.5)))
    fig.update_xaxes(type="category")
    items.append(("r5_line", estilo(fig, f"Escaneos por {nivel.lower()}", x=nivel, y="Escaneos")))

    rc = resumen(df, [col, "categoria_diag"])
    fig = px.bar(rc, x=col, y="escaneos", color="categoria_diag", barmode="stack", color_discrete_sequence=PALETA)
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(type="category")
    items.append(("r5_stack", estilo(fig, f"Diagnósticos por {nivel.lower()}", x=nivel, y="Escaneos")))

    pc = por.dropna(subset=["confianza"])
    if pc.empty:
        items.append(("r5_conf", sin_datos("Confianza promedio")))
    else:
        fig = px.line(pc, x=col, y="confianza", markers=True, color_discrete_sequence=[COLOR_GREEN_DEEP])
        fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME,
                                                      line=dict(color=COLOR_GREEN_DEEP, width=1.5)))
        fig.update_xaxes(type="category")
        items.append(("r5_conf", estilo(fig, f"Confianza promedio por {nivel.lower()} (%)", x=nivel, y="Confianza %")))

    tm = resumen(df.dropna(subset=["anio", "trimestre", "nombre_mes"]), ["anio", "trimestre", "nombre_mes"])
    tm = tm[tm["escaneos"] > 0]
    if tm.empty:
        items.append(("r5_tree", sin_datos("Jerarquía de tiempo")))
    else:
        fig = px.treemap(tm, path=["anio", "trimestre", "nombre_mes"], values="escaneos",
                         color_discrete_sequence=PALETA)
        fig.update_traces(textinfo="label+value",
                          textfont=dict(color="#ffffff", size=12),
                          marker=dict(line=dict(color="#ffffff", width=1)))
        items.append(("r5_tree", estilo(fig, "Año → trimestre → mes")))
    rejilla(items)

    hallazgo(f"El {nivel.lower()} con más escaneos es <b>{escape(str(pico[col]))}</b> ({fmt_int(pico['escaneos'])})."
             + (f" La mayor incidencia de plaga o enfermedad ocurre en <b>{escape(str(pico_inc.idxmax()))}</b> ({fmt_int(pico_inc.max())})."
                if pico_inc is not None else ""))


def pagina_r6(df):
    d = sin_desc(df, "modelo")
    if d.empty:
        st.info(sin_datos("R6"))
        return
    rv = resumen(d, "modelo")
    ra = resumen(sin_desc(d, "algoritmo"), "algoritmo")
    mejor_v, _ = _mejor_peor(rv)
    mejor_a, _ = _mejor_peor(ra)

    k = st.columns(4)
    kpi(k[0], "Algoritmos", str(ra["algoritmo"].nunique()),
        ", ".join(escape(str(a)) for a in ra["algoritmo"].head(3)), color=COLOR_GREEN)
    kpi(k[1], "Versiones", str(rv["modelo"].nunique()), "sin contar «desconocido»", color=COLOR_LIME)
    kpi(k[2], "Mejor algoritmo", escape(str(mejor_a["algoritmo"])) if mejor_a is not None else "s/d",
        fmt_pct(mejor_a["acierto_pct"]) if mejor_a is not None else "", color=COLOR_GREEN_SOFT)
    kpi(k[3], "Mejor versión", escape(str(mejor_v["modelo"])) if mejor_v is not None else "s/d",
        fmt_pct(mejor_v["acierto_pct"]) if mejor_v is not None else "", color=COLOR_GREEN_DEEP)

    items = []
    rap = ra.dropna(subset=["acierto_pct"])
    if rap.empty:
        items.append(("r6_alg", sin_datos("Algoritmos")))
    else:
        largo = rap.melt(id_vars="algoritmo", value_vars=["acierto_pct", "confianza", "baja_pct"],
                         var_name="métrica", value_name="valor")
        largo["métrica"] = largo["métrica"].map({"acierto_pct": "Exactitud",
                                                  "confianza": "Confianza promedio",
                                                  "baja_pct": "Confianza baja"})
        fig = px.bar(largo, x="algoritmo", y="valor", color="métrica", barmode="group", text_auto=".0f",
                     color_discrete_sequence=[COLOR_GREEN_DEEP, COLOR_LIME, COLOR_WARM])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=9))
        items.append(("r6_alg", estilo(fig, "Algoritmos: exactitud, confianza y confianza baja (%)", y="%")))

    rvv = rv.dropna(subset=["acierto_pct"]).merge(d[["modelo", "algoritmo"]].drop_duplicates(), on="modelo", how="left")
    if rvv.empty:
        items.append(("r6_ver", sin_datos("Versiones")))
    else:
        fig = barra_h(rvv, "acierto_pct", "modelo", "Exactitud por versión (%)", ".1f", "Exactitud %", color="algoritmo")
        fig.add_vline(x=META_EXACTITUD, line_dash="dash", line_color=COLOR_META, line_width=1.5)
        items.append(("r6_ver", fig))

    rs = rv.dropna(subset=["acierto_pct", "confianza"])
    if rs.empty:
        items.append(("r6_sc", sin_datos("Confianza vs. exactitud")))
    else:
        fig = px.scatter(rs, x="confianza", y="acierto_pct", size="escaneos", hover_name="modelo",
                         color_discrete_sequence=[COLOR_GREEN_DEEP])
        fig.update_traces(marker=dict(line=dict(width=2, color=COLOR_LIME)))
        items.append(("r6_sc", estilo(fig, "Confianza vs. exactitud por versión", x="Confianza %", y="Exactitud %")))

    rt = rv[rv["escaneos"] > 0]
    fig = px.treemap(rt, path=["modelo"], values="escaneos", color="modelo", color_discrete_sequence=PALETA)
    fig.update_traces(textinfo="label+value", textposition="top left",
                      textfont=dict(color="#ffffff", size=12),
                      marker=dict(line=dict(color="#ffffff", width=1)))
    items.append(("r6_tree", estilo(fig, "Escaneos por versión")))
    rejilla(items)

    if mejor_v is not None:
        texto = f"La versión con mayor exactitud es <b>{escape(str(mejor_v['modelo']))}</b> ({mejor_v['acierto_pct']:.1f}%)."
        if mejor_a is not None:
            texto += f" Entre algoritmos, destaca <b>{escape(str(mejor_a['algoritmo']))}</b> ({mejor_a['acierto_pct']:.1f}%)."
        texto += " Esta es la versión que conviene mantener en producción si cumple la meta."
        hallazgo(texto)


def pagina_decisiones(df):
    hall, recs = calcular_hallazgos(df)
    if not hall:
        st.info(sin_datos("Toma de decisiones"))
        return

    total = df["cantidad_escaneos"].sum()
    prec = precision(df)
    baja = pct_baja(df)
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()
    inc_pct = inc / total * 100 if total else np.nan
    sin_verif = df.loc[df["acierto"].isna(), "cantidad_escaneos"].sum() / total * 100 if total else np.nan

    seccion("01", "Planteamiento del análisis")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            "<div class='info-card'><div class='lbl'>¿Qué queremos analizar?</div>"
            "<div class='txt'>El desempeño del detector inteligente del gusano cogollero y la roya del maíz a lo "
            "largo del tiempo: exactitud del modelo, incidencia por parcela, confianza de los diagnósticos y "
            "confiabilidad de las versiones desplegadas.</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='info-card'><div class='lbl'>¿Qué buscamos?</div>"
            "<div class='txt'>Identificar en qué parcelas y periodos intervenir, qué versión del modelo mantener "
            "en producción y cuándo conviene reentrenar, con base en evidencia estadística y no en percepciones.</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Decisión a tomar</div>"
            "<div class='txt'>Determinar si el modelo actual cumple la meta de exactitud, qué acciones tomar sobre "
            "las parcelas con mayor incidencia y si es necesario reentrenar o cambiar de versión del algoritmo.</div></div>",
            unsafe_allow_html=True,
        )

    seccion("02", "Análisis estadístico")
    k = st.columns(4)
    kpi(k[0], "Exactitud global", fmt_pct(prec), f"meta ≥ {META_EXACTITUD}%", color=COLOR_GREEN)
    kpi(k[1], "Incidencia (plaga + enfermedad)", fmt_pct(inc_pct),
        f"{fmt_int(inc)} de {fmt_int(total)}", color=COLOR_LIME)
    kpi(k[2], "Escaneos con confianza baja", fmt_pct(baja), f"umbral < {UMBRAL}%", color=COLOR_GREEN_SOFT)
    kpi(k[3], "Escaneos sin verificar", fmt_pct(sin_verif), "sin clase real del experto", color=COLOR_GREEN_DEEP)

    st.markdown("**Indicadores de decisión**")
    st.table(tabla_decision(df))

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown("**Hallazgos clave**")
    for i in range(0, len(hall), 3):
        cols = st.columns(3, gap="medium")
        for col, (titulo, texto) in zip(cols, hall[i:i + 3]):
            col.markdown(
                f"<div class='card'><div class='card-t'>{titulo}</div><div class='card-x'>{texto}</div></div>",
                unsafe_allow_html=True,
            )

    seccion("03", "Decisión recomendada")
    if pd.isna(prec):
        principal = ("No hay datos verificados suficientes para calcular la exactitud del modelo. "
                     "Se recomienda completar la verificación del experto antes de tomar decisiones operativas.")
    elif prec >= META_EXACTITUD:
        principal = (f"El modelo <b>cumple</b> la meta de exactitud ({prec:.1f}% ≥ {META_EXACTITUD}%). "
                     "Se recomienda mantener la versión actual en producción y concentrar los esfuerzos en las "
                     "parcelas con mayor incidencia.")
    else:
        principal = (f"El modelo <b>no cumple</b> la meta de exactitud ({prec:.1f}% &lt; {META_EXACTITUD}%). "
                     "Se recomienda reentrenar antes de seguir usando el modelo como base para decisiones operativas.")
    st.markdown(f"<div class='reco-box'><b class='principal'>Decisión principal</b>{principal}</div>",
                unsafe_allow_html=True)

    if recs:
        for n, texto in enumerate(recs, 1):
            st.markdown(f"<div class='reco-box'><b>Acción {n}.</b> {texto}</div>", unsafe_allow_html=True)


PAGINAS_FN = {
    "inicio": pagina_inicio, "r1": pagina_r1, "r2": pagina_r2, "r3": pagina_r3,
    "r4": pagina_r4, "r5": pagina_r5, "r6": pagina_r6, "decisiones": pagina_decisiones,
}

# ===== CARGA Y LAYOUT =====
try:
    df_all = cargar_datos()
except FileNotFoundError as e:
    st.error(f"No encuentro el archivo {Path(str(e.filename)).name}. Verifica que la carpeta data/ tenga los 5 CSV.")
    st.stop()

if "pagina" not in st.session_state:
    st.session_state.pagina = PAGINAS[0][0]


def ir_a(clave):
    st.session_state.pagina = clave


topbar()

col_menu, col_main = st.columns([1.1, 4.8], gap="medium")

# Sidebar
with col_menu:
    with st.container(key="menu"):
        st.markdown(
            f"<div class='brand'><div class='brand-row'>"
            f"<div class='brand-logo'>{logo_html()}</div>"
            f"<div class='brand-text'>"
            f"<div class='brand-n'>{NOMBRE}</div>"
            f"<div class='brand-s'>{SUBTITULO}</div>"
            f"</div></div></div>",
            unsafe_allow_html=True,
        )
        for clave, icono, etiqueta, _ in PAGINAS:
            boton(f"{icono}    {etiqueta}", key=f"nav_{clave}",
                  type="primary" if st.session_state.pagina == clave else "secondary",
                  on_click=ir_a, args=(clave,))

        st.markdown("<div class='filtros-t'>Filtros</div>", unsafe_allow_html=True)
        f_anio = st.multiselect("Año", sorted(df_all["anio"].dropna().unique().tolist()), key="f_anio")
        f_trim = st.multiselect("Trimestre", sorted(df_all["trimestre"].dropna().unique().tolist()), key="f_trim")
        f_parc = st.multiselect("Parcela", sorted(sin_desc(df_all, "parcela")["parcela"].unique().tolist()), key="f_parc")

# Filtros aplicados
df = df_all
if f_anio:
    df = df[df["anio"].isin(f_anio)]
if f_trim:
    df = df[df["trimestre"].isin(f_trim)]
if f_parc:
    df = df[df["parcela"].isin(f_parc)]

# Mini stat al pie del sidebar (meta de exactitud)
prec_actual = precision(df) if not df.empty else np.nan
pct_actual = 0 if pd.isna(prec_actual) else max(0, min(100, prec_actual))
with col_menu:
    st.markdown(
        f"<div class='mini-stat'>"
        f"<div class='mini-stat-title'>Meta exactitud</div>"
        f"<div class='mini-stat-value'>{fmt_pct(prec_actual)}</div>"
        f"<div class='mini-stat-bar'><div class='mini-stat-fill' style='width:{pct_actual}%'></div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

titulos = {clave: titulo for clave, _, _, titulo in PAGINAS}
with col_main:
    encabezado(st.session_state.pagina, titulos[st.session_state.pagina])
    if df.empty:
        st.warning("No hay datos con los filtros seleccionados. Quita algún filtro en el menú.")
    else:
        PAGINAS_FN[st.session_state.pagina](df)
    st.markdown(
        "<div class='pie'>Fuente: PostgreSQL · db_detector_maiz (modelo ROLAP) · Instituto Tecnológico de Tizimín</div>",
        unsafe_allow_html=True,
    )
