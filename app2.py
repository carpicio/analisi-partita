# @title 💎 DASHBOARD V26: CONTROLLO MANUALE (Anti-Errore)
# @markdown 1. Premi Play ▶️.
# @markdown 2. Clicca il pulsante BLU "CARICA DATI" e attendi la conferma.
# @markdown 3. Seleziona le squadre dai menu che appariranno.
# @markdown 4. Clicca il pulsante VERDE "AVVIA ANALISI".

import sys
import subprocess
import os

# Installazione librerie
try:
    import lifelines
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines"])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. CONFIGURAZIONE E VARIABILI GLOBALI
# ==========================================
available_files = [f for f in os.listdir() if (f.endswith('.csv') or f.endswith('.xlsx')) and 'sample_data' not in f]
if not available_files: available_files = ['Nessun file trovato']

global_df = pd.DataFrame()

# ==========================================
# 2. FUNZIONI DI CARICAMENTO
# ==========================================
def load_dataset(nome_file):
    try:
        with open(nome_file, 'r', encoding='latin1', errors='replace') as f:
            lines = [f.readline() for _ in range(5)]
            sep = ';' if lines[0].count(';') > lines[0].count(',') else ','

        df_raw = pd.read_csv(nome_file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False, header=None)
        
        header = df_raw.iloc[0].astype(str).str.strip().str.upper().tolist()
        seen = {}
        unique_header = []
        for col in header:
            if col in seen:
                seen[col] += 1
                unique_header.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                unique_header.append(col)
                
        df = df_raw.iloc[1:].copy()
        df.columns = unique_header
        
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    found = next((c for c in df.columns if c == candidate), None)
                    if found:
                        df.rename(columns={found: target}, inplace=True)
                        break

        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 3. WIDGETS
# ==========================================
style = {'description_width': 'initial'}
layout = widgets.Layout(width='98%')

# Sezione 1: Caricamento
w_file = widgets.Dropdown(options=available_files, description='📂 FILE:', style=style, layout=layout)
btn_load = widgets.Button(description="🔄 1. CARICA DATI", button_style='info', layout=layout)

# Sezione 2: Selezione (Inizialmente vuoti/disabilitati)
w_paese = widgets.Dropdown(description='🌍 Paese:', style=style, layout=layout, disabled=True)
w_lega = widgets.Dropdown(description='🏆 Lega:', style=style, layout=layout, disabled=True)
w_home = widgets.Dropdown(description='🏠 Casa:', style=style, layout=layout, disabled=True)
w_away = widgets.Dropdown(description='✈️ Ospite:', style=style, layout=layout, disabled=True)

btn_run = widgets.Button(description="🚀 2. AVVIA ANALISI", button_style='success', layout=layout, disabled=True)

out_log = widgets.Output()
out_res = widgets.Output()

# ==========================================
# 4. LOGICA PULSANTI
# ==========================================

def on_load_click(b):
    global global_df
    with out_log:
        clear_output()
        print(f"⏳ Caricamento {w_file.value}...")
        global_df = load_dataset(w_file.value)
        
        if not global_df.empty:
            # Sblocca e popola widget
            paesi = sorted(global_df['PAESE'].unique()) if 'PAESE' in global_df.columns else []
            
            w_paese.options = paesi
            w_paese.disabled = False
            if paesi: w_paese.value = paesi[0]
            
            w_lega.disabled = False
            w_home.disabled = False
            w_away.disabled = False
            btn_run.disabled = False
            
            update_leghe() # Trigger primo aggiornamento a cascata
            print(f"✅ File caricato: {len(global_df)} righe. ORA SELEZIONA LE SQUADRE.")
        else:
            print("❌ Errore: File vuoto o illeggibile.")

def update_leghe(*args):
    if not global_df.empty and w_paese.value:
        leghe = sorted(global_df[global_df['PAESE'] == w_paese.value]['LEGA'].unique())
        w_lega.options = leghe
        if leghe: w_lega.value = leghe[0]
        update_squadre()

def update_squadre(*args):
    if not global_df.empty and w_paese.value and w_lega.value:
        mask = (global_df['PAESE'] == w_paese.value) & (global_df['LEGA'] == w_lega.value)
        teams = sorted(pd.concat([global_df[mask]['CASA'], global_df[mask]['OSPITE']]).unique())
        w_home.options = teams
        w_away.options = teams
        if len(teams) > 1:
            w_home.value = teams[0]
            w_away.value = teams[1]

w_paese.observe(update_leghe, 'value')
w_lega.observe(update_squadre, 'value')

def run_analysis(b):
    with out_res:
        clear_output()
        
        sel_p, sel_l = w_paese.value, w_lega.value
        sel_h, sel_a = w_home.value, w_away.value
        
        print(f"⚙️ ANALISI IN CORSO: {sel_h} vs {sel_a}...\n")
        
        df_league = global_df[(global_df['PAESE'] == sel_p) & (global_df['LEGA'] == sel_l)].copy()
        intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
        
        def get_minutes(val):
            if pd.isna(val): return []
            s = str(val).replace(',', '.').replace(';', ' ')
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            res = []
            for x in nums:
                try:
                    n = int(float(x))
                    if 0 <= n <= 130: res.append(n)
                except: pass
            return res

        c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else 'GOALMINCASA'
        c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else 'GOALMINOSPITE'

        goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
        goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
        match_h, match_a = 0, 0
        times_h, times_a, times_league = [], [], []
        stats_match = {sel_h: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
                       sel_a: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}}

        for _, row in df_league.iterrows():
            h, a = row['CASA'], row['OSPITE']
            min_h = get_minutes(row.get(c_h))
            min_a = get_minutes(row.get(c_a))
            
            # Media Lega
            if min_h: times_league.append(min(min_h))
            if min_a: times_league.append(min(min_a))

            # Heatmap
            if h in stats_match:
                for m in min_h:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    stats_match[h]['F'][intervals[idx]] += 1
                for m in min_a:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    stats_match[h]['S'][intervals[idx]] += 1
            
            if a in stats_match:
                for m in min_a:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    stats_match[a]['F'][intervals[idx]] += 1
                for m in min_h:
                    idx = min(5, (m-1)//15)
                    if m > 45 and m <= 60 and idx < 3: idx = 3
                    stats_match[a]['S'][intervals[idx]] += 1

            # Stats
            if h == sel_h:
                match_h += 1
                goals_h['FT'] += len(min_h)
                goals_h['HT'] += len([x for x in min_h if x <= 45])
                goals_h['S_FT'] += len(min_a)
                goals_h['S_HT'] += len([x for x in min_a if x <= 45])
                if min_h: times_h.append(min(min_h))
            
            if a == sel_a:
                match_a += 1
                goals_a['FT'] += len(min_a)
                goals_a['HT'] += len([x for x in min_a if x <= 45])
                goals_a['S_FT'] += len(min_h)
                goals_a['S_HT'] += len([x for x in min_h if x <= 45])
                if min_a: times_a.append(min(min_a))

        # Calcoli Medie
        def safe_div(n, d): return n / d if d > 0 else 0
        avg_h_ft = safe_div(goals_h['FT'], match_h)
        avg_h_ht = safe_div(goals_h['HT'], match_h)
        avg_h_conc_ft = safe_div(goals_h['S_FT'], match_h)
        avg_h_conc_ht = safe_div(goals_h['S_HT'], match_h)
        avg_a_ft = safe_div(goals_a['FT'], match_a)
        avg_a_ht = safe_div(goals_a['HT'], match_a)
        avg_a_conc_ft = safe_div(goals_a['S_FT'], match_a)
        avg_a_conc_ht = safe_div(goals_a['S_HT'], match_a)

        # Poisson
        exp_h_ft = (avg_h_ft + avg_a_conc_ft) / 2
        exp_a_ft = (avg_a_ft + avg_h_conc_ft) / 2
        exp_h_ht = (avg_h_ht + avg_a_conc_ht) / 2
        exp_a_ht = (avg_a_ht + avg_h_conc_ht) / 2

        def calc_poisson(lam_h, lam_a):
            probs = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            return np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1))

        p1_ft, px_ft, p2_ft = calc_poisson(exp_h_ft, exp_a_ft)
        prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
        prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))

        print(f"\n📊 STATISTICHE MEDIE (Casa vs Fuori)")
        print(f"{sel_h:<20} | Fatti: {avg_h_ht:.2f} (HT) - {avg_h_ft:.2f} (FT) | Subiti: {avg_h_conc_ht:.2f} (HT) - {avg_h_conc_ft:.2f} (FT)")
        print(f"{sel_a:<20} | Fatti: {avg_a_ht:.2f} (HT) - {avg_a_ft:.2f} (FT) | Subiti: {avg_a_conc_ht:.2f} (HT) - {avg_a_conc_ft:.2f} (FT)")

        print(f"\n🎲 PREVISIONI POISSON")
        print(f"   1° TEMPO:  1 ({calc_poisson(exp_h_ht, exp_a_ht)[0]*100:.1f}%)  X ({calc_poisson(exp_h_ht, exp_a_ht)[1]*100:.1f}%)  2 ({calc_poisson(exp_h_ht, exp_a_ht)[2]*100:.1f}%)")
        print(f"   FINALE:    1 ({p1_ft*100:.1f}%)  X ({px_ft*100:.1f}%)  2 ({p2_ft*100:.1f}%)")
        print(f"   SPECIFICHE HT: 0-0 ({prob_00_ht*100:.1f}%) | Under 1.5 ({prob_u15_ht*100:.1f}%)")

        # Grafici
        rows_f, rows_s = [], []
        for t in [sel_h, sel_a]:
            d = stats_match[t]
            rows_f.append({**{'SQUADRA': t}, **d['F']})
            rows_s.append({**{'SQUADRA': t}, **d['S']})
        
        df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
        df_s = pd.DataFrame(rows_s).set_index('SQUADRA')

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=axes[0])
        axes[0].set_title(f'⚽ DENSITÀ GOL FATTI', fontweight='bold')
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=axes[1])
        axes[1].set_title(f'🛡️ DENSITÀ GOL SUBITI', fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Kaplan-Meier
        plt.figure(figsize=(10, 5))
        kmf_h = KaplanMeierFitter()
        kmf_a = KaplanMeierFitter()
        kmf_l = KaplanMeierFitter()
        
        if times_h and times_a:
            kmf_h.fit(times_h, label=f'{sel_h} Gol')
            kmf_a.fit(times_a, label=f'{sel_a} Gol')
            
            if times_league:
                kmf_l.fit(times_league, label='Media Lega')
                kmf_l.plot_survival_function(ax=plt.gca(), ci_show=False, linewidth=2, color='gray', linestyle='--')

            ax = plt.gca()
            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.title('📉 RITMO GOL (Tempo al 1° Gol)')
            plt.grid(True, alpha=0.3)
            plt.axvline(45, color='green', linestyle='--')
            plt.legend()
            plt.show()
        else:
            print("⚠️ Dati insufficienti per Kaplan-Meier.")

# Collegamento Pulsanti
btn_load.on_click(on_load_click)
btn_run.on_click(run_analysis)

# Layout Finale
box_1 = widgets.VBox([
    widgets.Label("1. CARICA FILE:"), w_file, btn_load, out_log
])
box_2 = widgets.VBox([
    widgets.Label("2. SELEZIONA MATCH:"), 
    widgets.HBox([w_paese, w_lega]),
    widgets.HBox([w_home, w_away]),
    btn_run
])

display(box_1, box_2, out_res)
