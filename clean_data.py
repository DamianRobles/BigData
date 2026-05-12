"""
clean_data.py
Consolida y limpia los 47 CSVs de transacciones inmobiliarias MLIT Japón.
Input : data/raw/01.csv ... 47.csv  +  data/raw/index.csv
Output: data/clean/japan_clean.csv

Estrategia de memoria: escritura incremental — procesa un CSV a la vez
y hace append al archivo de salida. Nunca carga más de 1 CSV en RAM.
El CSV más grande tiene 406,575 filas → ~84k tras filtros → seguro en memoria.
"""

import time
import pandas as pd
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TYPE  = "Residential Land(Land and Building)"
MAX_AGE      = 80     # años — eliminar antigüedades imposibles
PRICE_Q_LOW  = 0.01   # percentil inferior outliers de precio
PRICE_Q_HIGH = 0.99   # percentil superior outliers de precio

KEEP_COLS = [
    'Prefecture', 'Municipality', 'MunicipalityCode',
    'TradePrice', 'Area', 'BuildingAge', 'BuildingYear', 'Year', 'Quarter',
    'TimeToNearestStation', 'Structure', 'CityPlanning',
    'CoverageRatio', 'FloorAreaRatio', 'LandShape', 'PrewarBuilding'
]

REQUIRED_FEATURES = [
    'TradePrice', 'BuildingAge', 'Area',
    'TimeToNearestStation', 'Structure', 'CityPlanning'
]


def clean_prefecture(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las reglas de limpieza a un DataFrame de una prefectura."""

    # 1. Filtrar tipo de transacción
    df = df[df['Type'] == TARGET_TYPE].copy()
    if df.empty:
        return df

    # 2. Eliminar registros sin BuildingYear
    df = df[df['BuildingYear'].notna()]
    if df.empty:
        return df
    df['BuildingYear'] = df['BuildingYear'].astype(int)

    # 3. Calcular BuildingAge y filtrar valores imposibles
    df['BuildingAge'] = df['Year'] - df['BuildingYear']
    df = df[(df['BuildingAge'] >= 0) & (df['BuildingAge'] <= MAX_AGE)]

    # 4. Eliminar registros con área censurada (exactitud del dato requerida)
    if 'AreaIsGreaterFlag' in df.columns:
        df = df[df['AreaIsGreaterFlag'] == 0]

    # TimeToNearestStation puede venir como texto ("30-60", "1H30") → numérico
    df['TimeToNearestStation'] = pd.to_numeric(
        df['TimeToNearestStation'], errors='coerce'
    )

    # 5. Limpiar Structure: quedarse solo con el material dominante
    #    Ej: "W, LS" → "W"  |  "RC, S" → "RC"
    df['Structure'] = df['Structure'].astype(str).str.split(',').str[0].str.strip()
    df.loc[df['Structure'].isin(['nan', 'None', '']), 'Structure'] = pd.NA

    # 6. Eliminar filas con nulos en las columnas requeridas por el modelo
    df = df.dropna(subset=REQUIRED_FEATURES)
    if df.empty:
        return df

    # 7. Filtrar outliers de precio dentro de cada prefectura
    low  = df['TradePrice'].quantile(PRICE_Q_LOW)
    high = df['TradePrice'].quantile(PRICE_Q_HIGH)
    df = df[(df['TradePrice'] >= low) & (df['TradePrice'] <= high)]

    # 8. Conservar solo las columnas necesarias (las que existan)
    cols_present = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols_present]

    return df


def main() -> None:
    # ── Inicio del cronómetro total ──────────────────────────────────────────
    start_total = time.time()

    # Verificar que exista el catálogo de prefecturas
    index_path = RAW_DIR / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"No se encontró {index_path}")

    index_df = pd.read_csv(index_path)
    print(f"Catálogo cargado: {len(index_df)} prefecturas")

    # Archivo de salida: escribir encabezado una sola vez
    output_path = CLEAN_DIR / "japan_clean.csv"
    if output_path.exists():
        output_path.unlink()  # empezar desde cero en cada ejecución

    write_header = True
    total_rows = 0
    total_original = 0

    # Procesar cada CSV de forma incremental
    for num in range(1, 48):
        filepath = RAW_DIR / f"{num:02d}.csv"
        if not filepath.exists():
            print(f"[SKIP] {filepath} no encontrado")
            continue

        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"[ERROR] {filepath.name}: {e}")
            continue

        rows_original = len(df)
        total_original += rows_original
        df_clean = clean_prefecture(df)

        if df_clean.empty:
            print(f"[WARN] {filepath.name}: sin registros tras limpieza")
            continue

        df_clean.to_csv(output_path, mode='a', header=write_header, index=False)
        write_header = False
        total_rows += len(df_clean)

        pct = len(df_clean) / rows_original * 100 if rows_original else 0
        print(
            f"[OK] {filepath.name:8s}: "
            f"{rows_original:>8,} originales → {len(df_clean):>6,} limpias "
            f"({pct:5.1f}%)"
        )

    # ── Métricas de velocidad ────────────────────────────────────────────────
    elapsed_total = time.time() - start_total

    print(f"\n✓ Dataset limpio guardado en: {output_path}")
    print(f"  Total registros limpios : {total_rows:,}")
    print(f"  Total registros crudos  : {total_original:,}")
    print(f"  Tiempo total            : {elapsed_total:.1f} segundos")
    print(f"  Velocidad (filas crudas): {total_original / elapsed_total:,.0f} filas/segundo")
    print(f"  Velocidad (filas limpias): {total_rows / elapsed_total:,.0f} filas/segundo")

    if total_rows == 0:
        print("  ⚠ El archivo de salida está vacío — revisar filtros.")
        return

    # Verificación rápida leyendo solo las primeras filas
    sample = pd.read_csv(output_path, nrows=5)
    print(f"  Columnas: {list(sample.columns)}")
    print(
        f"  Rango BuildingAge en muestra: "
        f"{sample['BuildingAge'].min()}–{sample['BuildingAge'].max()}"
    )


if __name__ == "__main__":
    main()