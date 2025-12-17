import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Union
from io import BytesIO

def clean_currency(x: Union[str, float]) -> float:
    if isinstance(x, str):
        # If a comma exists, assume it's a German-style decimal separator
        if ',' in x:
            x = x.replace('.', '').replace(',', '.')
    return float(x)

def load_and_prep_data(uploaded_file: BytesIO, selected_col: str) -> pd.DataFrame:
    # WICHTIG: Pointer auf Anfang setzen, falls die Datei vorher schon berührt wurde
    uploaded_file.seek(0)
    
    # 1. Trennzeichen (Delimiter) automatisch erkennen
    # Wir lesen die erste Zeile, um zu prüfen, ob ';' oder ',' häufiger vorkommt
    try:
        sample_line = uploaded_file.readline().decode('utf-8')
    except UnicodeDecodeError:
        # Fallback auf Latin-1, falls UTF-8 fehlschlägt (z.B. Excel-CSVs)
        uploaded_file.seek(0)
        sample_line = uploaded_file.readline().decode('latin-1')

    # Entscheidung: Welches Trennzeichen?
    delimiter = ';' if sample_line.count(';') > sample_line.count(',') else ','
    
    # Pointer wieder zurücksetzen für das eigentliche Laden mit Pandas
    uploaded_file.seek(0)
    
    # 2. Datei mit dem erkannten Trennzeichen laden
    df = pd.read_csv(uploaded_file, sep=delimiter)
    
    # 3. Die erste Spalte standardisieren (Anforderung 1)
    # Wir benennen die erste Spalte (Index 0) intern zu 'Datum' um.
    first_col_name = df.columns[0]
    df.rename(columns={first_col_name: 'datum'}, inplace=True)
    
    # Sicherheitscheck: Existiert die ausgewählte Spalte überhaupt?
    # (Falls der User eine Spalte wählt, die durch falsches Parsing nicht da ist)
    if selected_col not in df.columns:
        # Falls die Spalte nicht gefunden wird, nehmen wir die zweite Spalte als Fallback
        selected_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Bereinigung der Zahlenwerte (Falls String)
    if df[selected_col].dtype == 'object':
         # clean_currency Logik direkt hier integriert oder deine Funktion nutzen
         # Annahme: Deine clean_currency Funktion existiert global. 
         # Falls nicht, hier eine robuste Inline-Lösung:
         df[selected_col] = df[selected_col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    
    # 4. Datum konvertieren
    # errors='coerce' verhindert Abstürze bei fehlerhaften Datumszeilen
    df['datum'] = pd.to_datetime(df['datum'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['datum']).sort_values('datum')
    
    # Nullen entfernen
    df_clean = df[df[selected_col] > 0].copy()
    
    # Returns berechnen
    df_clean['LogReturns'] = np.log(df_clean[selected_col] / df_clean[selected_col].shift(1))
    df_clean['SimpleReturns'] = df_clean[selected_col].pct_change()
    
    return df_clean.dropna()

def run_simulation_arima(df: pd.DataFrame, col_name: str, horizon: int, n_sims: int) -> np.ndarray:
    # 1. Fit ARIMA (No Drift)
    model = ARIMA(df['LogReturns'], order=(1, 0, 1), trend='n')
    model_res = model.fit()
    
    # 2. Setup
    params = model_res.params
    ar = params.get('ar.L1', 0)
    ma = params.get('ma.L1', 0)
    residuals = model_res.resid.values - np.mean(model_res.resid.values)
    
    last_price = df[col_name].iloc[-1]
    last_return = df['LogReturns'].iloc[-1]
    last_resid = residuals[-1]
    
    sim_paths = np.zeros((n_sims, horizon))
    
    # 3. Simulation Loop
    for i in range(n_sims):
        path_y = [last_return]
        path_eps = [last_resid]
        current_path = []
        
        shocks = np.random.choice(residuals, size=horizon, replace=True)
        
        for t in range(horizon):
            eps_t = shocks[t]
            y_t = (ar * path_y[-1]) + (ma * path_eps[-1]) + eps_t
            current_path.append(y_t)
            path_y.append(y_t)
            path_eps.append(eps_t)
            
        sim_paths[i, :] = np.array(current_path)
    
    # 4. Rekonstruktion
    cum_ret = np.cumsum(sim_paths, axis=1)
    price_paths = last_price * np.exp(cum_ret)
    return price_paths

def run_simulation_block(df: pd.DataFrame, col_name: str, horizon: int, n_sims: int, block_size: int = 6) -> np.ndarray:
    returns = df['LogReturns'].values # Wir nehmen LogReturns für Konsistenz
    n = len(returns)
    last_price = df[col_name].iloc[-1]
    
    blocks = [returns[i:i+block_size] for i in range(n - block_size + 1)]
    
    sim_paths = np.zeros((n_sims, horizon))
    
    for i in range(n_sims):
        path = []
        while len(path) < horizon:
            path.extend(blocks[np.random.randint(0, len(blocks))])
        sim_paths[i, :] = path[:horizon]
        
    cum_ret = np.cumsum(sim_paths, axis=1)
    price_paths = last_price * np.exp(cum_ret)
    return price_paths