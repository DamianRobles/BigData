"""
make_figures.py
Genera las 4 figuras solicitadas para la tesis a partir de
data/clean/japan_clean.csv y las guarda en figures/.

  1. Curva de depreciación: precio predicho vs BuildingAge por Structure
  2. Scatter de residuos del modelo OLS (fitted vs residuals)
  3. Bar chart top 15 prefecturas por precio/m² del land value floor
  4. Heatmap: precio mediano por prefectura × año
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

CLEAN_CSV = Path("data/clean/japan_clean.csv")
FIG_DIR   = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")

# ── Cargar datos ─────────────────────────────────────────────────────────────
print("Cargando datos…")
df = pd.read_csv(CLEAN_CSV)
print(f"  {len(df):,} registros, {df['Prefecture'].nunique()} prefecturas")

# ── Feature engineering ──────────────────────────────────────────────────────
df["log_price"] = np.log(df["TradePrice"])
df_model = pd.get_dummies(
    df,
    columns=["Structure", "CityPlanning", "Prefecture"],
    drop_first=True,
)
feature_cols = (
    ["BuildingAge", "Area", "TimeToNearestStation"]
    + [c for c in df_model.columns if c.startswith("Structure_")]
    + [c for c in df_model.columns if c.startswith("CityPlanning_")]
    + [c for c in df_model.columns if c.startswith("Prefecture_")]
)
X = df_model[feature_cols].astype(float)
y = df_model["log_price"]

print(f"Ajustando regresión lineal ({len(feature_cols)} features)…")
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred
print(f"  R² (in-sample): {model.score(X, y):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Figura 1 — Curva de depreciación por tipo de estructura
# ─────────────────────────────────────────────────────────────────────────────
print("[1/4] Curva de depreciación por estructura…")

ages = np.arange(0, 81)
structures = df["Structure"].value_counts().index[:6].tolist()
median_row = X.median()

fig, ax = plt.subplots(figsize=(12, 7))
palette = sns.color_palette("tab10", n_colors=len(structures))

for color, struct in zip(palette, structures):
    X_sim = pd.DataFrame([median_row.values] * len(ages), columns=X.columns)
    X_sim["BuildingAge"] = ages
    for c in X_sim.columns:
        if c.startswith("Structure_"):
            X_sim[c] = 0.0
    dummy_col = f"Structure_{struct}"
    if dummy_col in X_sim.columns:
        X_sim[dummy_col] = 1.0
    price_pred = np.exp(model.predict(X_sim))
    ax.plot(ages, price_pred / 1e6, label=struct, linewidth=2.2, color=color)

ax.set_xlabel("Antigüedad del inmueble (años)")
ax.set_ylabel("Precio predicho (millones de JPY)")
ax.set_title(
    "Curva de depreciación por tipo de estructura\n"
    "Resto de features fijadas en la mediana"
)
ax.legend(title="Structure", loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_depreciacion_por_estructura.png", dpi=140, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figura 2 — Residuos del modelo OLS
# ─────────────────────────────────────────────────────────────────────────────
print("[2/4] Scatter de residuos…")

# Submuestrear para que el scatter sea legible
SAMPLE = min(8000, len(df))
rng = np.random.default_rng(42)
idx = rng.choice(len(y_pred), size=SAMPLE, replace=False)

fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(y_pred[idx], residuals.iloc[idx], s=8, alpha=0.25, color="steelblue")
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Valor predicho — log(TradePrice)")
ax.set_ylabel("Residuo (y − ŷ)")
ax.set_title(
    f"Residuos del modelo OLS (muestra n={SAMPLE:,} de {len(df):,})\n"
    f"R² = {model.score(X, y):.4f}"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_residuos_ols.png", dpi=140, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figura 3 — Top 15 prefecturas por precio/m² del land value floor
# ─────────────────────────────────────────────────────────────────────────────
print("[3/4] Top 15 prefecturas por precio/m²…")

old_props = df[df["BuildingAge"] >= 50]
land_floor_mun = (
    old_props.groupby(["Prefecture", "Municipality"])
    .agg(median_price=("TradePrice", "median"),
         count=("TradePrice", "count"),
         median_area=("Area", "median"))
    .reset_index()
)
land_floor_mun["price_per_m2"] = (
    land_floor_mun["median_price"] / land_floor_mun["median_area"]
)
land_floor_mun = land_floor_mun[land_floor_mun["count"] >= 10]

land_floor_pref = (
    land_floor_mun.groupby("Prefecture")["price_per_m2"]
    .median()
    .sort_values(ascending=False)
)
top15 = land_floor_pref.head(15).iloc[::-1]   # invertir para barh

fig, ax = plt.subplots(figsize=(11, 8))
norm = plt.Normalize(top15.min(), top15.max())
colors = sns.color_palette("rocket_r", n_colors=len(top15))
ax.barh(top15.index, top15.values / 1000, color=colors, edgecolor="white")

for i, v in enumerate(top15.values / 1000):
    ax.text(v, i, f" {v:,.1f}k", va="center", fontsize=9)

ax.set_xlabel("Precio mediano por m² (miles de JPY)")
ax.set_title(
    "Top 15 prefecturas por valor del suelo\n"
    "(precio/m² mediano en inmuebles ≥50 años)"
)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_top15_prefecturas.png", dpi=140, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figura 4 — Heatmap: precio mediano por prefectura × año
# ─────────────────────────────────────────────────────────────────────────────
print("[4/4] Heatmap precio mediano por prefectura × año…")

heat = (
    df.groupby(["Prefecture", "Year"])["TradePrice"]
    .median()
    .unstack("Year")
    / 1e6
)
# Orden por mediana global descendente para que las prefecturas más caras queden arriba
heat = heat.loc[heat.median(axis=1).sort_values(ascending=False).index]

# Cap al percentil 95 para que outliers puntuales (p.ej. Hokkaido 2007)
# no aplasten la escala de color y oculten las diferencias entre el resto.
vmax = np.nanpercentile(heat.values, 95)

fig, ax = plt.subplots(figsize=(14, 14))
sns.heatmap(
    heat,
    cmap="rocket_r",
    vmin=0,
    vmax=vmax,
    cbar_kws={"label": f"Precio mediano (millones JPY) — escala capada al p95 = {vmax:.1f}"},
    linewidths=0.3,
    linecolor="white",
    ax=ax,
)
ax.set_title("Mapa de calor — precio mediano por prefectura y año")
ax.set_xlabel("Año")
ax.set_ylabel("Prefectura (orden por mediana global)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_heatmap_precio_prefectura.png", dpi=140, bbox_inches="tight")
plt.close()

print(f"\nOK - 4 figuras guardadas en {FIG_DIR.resolve()}")
for f in sorted(FIG_DIR.glob("fig*.png")):
    print(f"  - {f.name}")
