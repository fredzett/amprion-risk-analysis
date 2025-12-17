import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from logic import load_and_prep_data, run_simulation_arima, run_simulation_block

# --- SEITEN-KONFIGURATION ---
st.set_page_config(
    page_title="Plattform für Risikoanalyse & Prognose",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONELLES CSS ---
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #34495e;
        color: #ecf0f1;
        border: none;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIK ---


from typing import Union
# --- VISUALISIERUNG ---

def plot_historical_chart(df: pd.DataFrame, target_col: str) -> go.Figure:
    # Data is already sorted chronologically (ascending) from load_and_prep_data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['datum'], y=df[target_col], mode='lines', name='Historie', line=dict(color='#2c3e50', width=2)))
    fig.update_layout(
        title=dict(text=f"Historische Zeitreihe: {target_col}", font=dict(size=18, color='#2c3e50')),
        xaxis=dict(title="Datum", showgrid=True, gridcolor='#ecf0f1'),
        yaxis=dict(title="Wert", showgrid=True, gridcolor='#ecf0f1', rangemode='tozero'),
        template="plotly_white", margin=dict(l=40, r=40, t=50, b=40), height=400
    )
    return fig

def plot_forecast_chart(df: pd.DataFrame, paths: np.ndarray, target_col: str, title: str, color_hex: str, show_ci_80: bool, show_ci_95: bool) -> go.Figure:
    last_date = df['datum'].iloc[-1] # Data is sorted ascending, so last element is newest
    future_dates = pd.date_range(last_date, periods=paths.shape[1]+1, freq='MS')[1:]
    
    median_path = np.median(paths, axis=0)
    p10, p90 = np.percentile(paths, 10, axis=0), np.percentile(paths, 90, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    fig = go.Figure()
    # Anschluss an Historie - take the last 24 items which are the newest
    hist_subset = df.tail(24) 
    fig.add_trace(go.Scatter(x=hist_subset['datum'], y=hist_subset[target_col], mode='lines', name='Historie (Auszug)', line=dict(color='#2c3e50', width=2)))

    if show_ci_80:
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill='toself', fillcolor=color_hex, line=dict(color='rgba(0,0,0,0)'),
            opacity=0.2, name='80% Konfidenzintervall'
        ))
    
    fig.add_trace(go.Scatter(x=future_dates, y=median_path, mode='lines', name='Median Prognose', line=dict(color=color_hex, width=3)))
    
    if show_ci_95:
        fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', name='95. Perzentil (Risiko)', line=dict(color=color_hex, width=1.5, dash='dash')))

    fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='#2c3e50')),
            xaxis=dict(title="Datum", showgrid=True, gridcolor='#ecf0f1'),
            yaxis=dict(title="Prognosewert", showgrid=True, gridcolor='#ecf0f1', rangemode='tozero'),
            template="plotly_white", 
            # --- KORREKTUR START ---
            legend=dict(
                orientation="h",    # Horizontal
                yanchor="bottom",   # Den Boden der Legende als Anker nehmen
                y=1.02,             # Knapp über dem Plot
                xanchor="right",    # Den rechten Rand der Legende als Anker nehmen
                x=1                 # Am rechten Rand des Plots ausrichten
            ), 
            # --- KORREKTUR ENDE ---
            margin=dict(l=40, r=40, t=80, b=40), 
            height=500
        )
    return fig

# --- HAUPT-LAYOUT ---

st.title("Risiko- und Prognoseplattform für Zeitreihen")
st.markdown("DRAFT - NUR FÜR DEMOZWECKE")
st.divider()

with st.sidebar:
    st.header("Konfiguration")
    st.markdown("### 1. Datenimport")
    uploaded_file = st.file_uploader("Zeitreihen-CSV hochladen", type=["csv"])
    
    df_clean, target_col = pd.DataFrame(), None
    
    if uploaded_file:
        try:
            uploaded_file.seek(0)
            # Use a dummy delimiter that is unlikely to be in the header to read only the first row for column names
            df_raw = pd.read_csv(uploaded_file, delimiter=';', nrows=1)
            uploaded_file.seek(0)
            exclude_terms = ['datum', 'date', 'zeit', 'time']
            numeric_cols = [c for c in df_raw.columns if not any(term in c.lower() for term in exclude_terms)]            
            
            st.markdown("### 2. Modellparameter")
            target_col = st.selectbox("Zielvariable", numeric_cols)
            horizon = st.number_input("Prognosehorizont (Monate)", 6, 60, 24)
            n_sims = st.number_input("Iterationen", 1000, 10000, 1000, step=1000)
            block_size = st.number_input("Blockgröße (Monate)", 1, 24, 6)
            
            st.markdown("### 3. Output")
            show_ci_80 = st.checkbox("80% Konfidenzintervall", value=True)
            show_ci_95 = st.checkbox("95. Perzentil (Risiko)", value=True)
            st.markdown("---")
            run_simulation = st.button("Simulation ausführen")
        except Exception as e:
            st.error(f"Dateifehler oder Parameterproblem: {e}")

if uploaded_file and target_col:
    df_clean = load_and_prep_data(uploaded_file, target_col)
    if not df_clean.empty:
        st.subheader("Datenbasis")
        st.plotly_chart(plot_historical_chart(df_clean, target_col), use_container_width=True)
        
        if 'run_simulation' in locals() and run_simulation:
            st.divider()
            st.subheader("Prognoseergebnisse")
            with st.spinner("Berechne..."):
                paths_block = run_simulation_block(df_clean, target_col, horizon, n_sims, block_size)
                try:
                    paths_arima = run_simulation_arima(df_clean, target_col, horizon, n_sims)
                except Exception as e:
                    paths_arima = None
                    st.warning(f"ARIMA-Modell konnte nicht berechnet werden: {e}")
                
                def german_format(val: Union[float, int]) -> str:
                    # Formats a float into a German currency string (e.g., 1.234,56)
                    return f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                
                tab_arima, tab_block = st.tabs(["Parametrisch (ARIMA)", "Nicht-Parametrisch (Block Bootstrap)"])
                
                with tab_arima:
                    st.markdown(f"**ARIMA Residual Bootstrap**")
                    
                    if paths_arima is not None:
                        # Kennzahlen für letzten Prognosepunkt
                        median_arima = np.median(paths_arima, axis=0)[-1]
                        p80_arima = np.percentile(paths_arima, 80, axis=0)[-1]
                        p95_arima = np.percentile(paths_arima, 95, axis=0)[-1]
                        
                        st.markdown("##### Kennzahlen")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Letzter Act. Wert", german_format(df_clean[target_col].iloc[-1]))
                        col2.metric("Mittelwert", german_format(np.mean(paths_arima)))
                        col3.metric("Median", german_format(median_arima))
                        col4.metric("P80", german_format(p80_arima))
                        col5.metric("P95", german_format(p95_arima))

                        st.plotly_chart(plot_forecast_chart(df_clean, paths_arima, target_col, "Strukturelle Prognose", "#4a148c", show_ci_80, show_ci_95), use_container_width=True)
                    else:
                        st.warning("ARIMA-Modell konnte nicht berechnet werden.")

                with tab_block:
                    st.markdown(f"**Block Bootstrap** (Blockgröße: {block_size} M)")
                    
                    if paths_block is not None:
                        # Kennzahlen für letzten Prognosepunkt
                        median_block = np.median(paths_block, axis=0)[-1]
                        p80_block = np.percentile(paths_block, 80, axis=0)[-1]
                        p95_block = np.percentile(paths_block, 95, axis=0)[-1]

                        st.markdown("##### Kennzahlen")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Letzter Act. Wert", german_format(df_clean[target_col].iloc[-1]))
                        col2.metric("Mittelwert", german_format(np.mean(paths_block)))
                        col3.metric("Median", german_format(median_block))
                        col4.metric("P80", german_format(p80_block))
                        col5.metric("P95", german_format(p95_block))

                        st.plotly_chart(plot_forecast_chart(df_clean, paths_block, target_col, "Stresstest-Szenario", "#e65100", show_ci_80, show_ci_95), use_container_width=True)

                # --- CSV EXPORT LOGIK (NEU) ---
                dates = pd.date_range(df_clean['datum'].iloc[-1], periods=horizon+1, freq='MS')[1:]
                
                # 1. Historie aufbereiten
                df_hist = df_clean[['datum', target_col]].copy()
                df_hist.columns = ['Datum', 'Historie']
                
                # 2. Prognose-Daten aufbereiten
                df_forecast = pd.DataFrame({'Datum': dates})
                
                if paths_arima is not None:
                    # ARIMA Stats
                    df_forecast['ARIMA_Median'] = np.median(paths_arima, axis=0)
                    df_forecast['ARIMA_Lower80'] = np.percentile(paths_arima, 10, axis=0) # Untere Grenze
                    df_forecast['ARIMA_Upper80'] = np.percentile(paths_arima, 90, axis=0) # Obere Grenze
                    df_forecast['ARIMA_P95'] = np.percentile(paths_arima, 95, axis=0)
                
                if paths_block is not None:
                    # Block Stats
                    df_forecast['Block_Median'] = np.median(paths_block, axis=0)
                    df_forecast['Block_Lower80'] = np.percentile(paths_block, 10, axis=0)
                    df_forecast['Block_Upper80'] = np.percentile(paths_block, 90, axis=0)
                    df_forecast['Block_P95'] = np.percentile(paths_block, 95, axis=0)
                
                # 3. Zusammenführen (Historie + Prognose)
                df_export = pd.merge(df_hist, df_forecast, on='Datum', how='outer').sort_values('Datum', ascending=False)
                
                # 4. Speichern mit deutschem Format (Semikolon und Komma)
                csv_data = df_export.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig').encode('utf-8-sig')
                
                st.download_button("Vollständigen Datensatz herunterladen (CSV)", csv_data, f"forecast_full_{target_col}.csv", "text/csv")
    else: st.warning("Keine validen Daten.")
elif not uploaded_file: st.info("Bitte Datei hochladen.")