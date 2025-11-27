# @title 💎 DASHBOARD V24: FIX FINALE (Crash-Proof)
# @markdown 1. Premi Play ▶️.
# @markdown 2. Attendi il caricamento dei menu.
# @markdown 3. Seleziona tutto e premi AVVIA.

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
# 1. GESTIONE FILE
# ==========================================
available_files = [f for f in os.listdir() if (f.endswith('.csv') or f.endswith('.xlsx')) and 'sample_data' not in f]
if not available_files: available_files = ['Nessun file trovato']

global_df = pd.DataFrame()

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
# 2. WIDGETS
# ==========================================
style = {'description_width': 'initial'}
layout = widgets.Layout(width='98%')

w_file = widgets.Dropdown(options=available_files, description='📂 FILE:', style=style, layout=layout)
w_paese = widgets.Dropdown(description='1. 🌍 Paese:', style=style, layout=layout)
w_lega = widgets.Dropdown(description='2. 🏆 Lega:', style=style, layout=layout)
w_home = widgets.Dropdown(description='3. 🏠 Casa:', style=style, layout=layout)
w_away = widgets.Dropdown(description='4. ✈️ Ospite:', style=style, layout=layout)

btn_run = widgets.Button(description="🚀 AVVIA ANALISI", button_style='primary', layout=layout)
out = widgets.Output()

def on_file_change(change):
    global global_df
    if change['new'] and change['new'] != 'Nessun file trovato':
        with out:
            # clear_output()
            print(f"⏳ Caricamento {change['new']}...")
            global_df = load_dataset(change['new'])
            if not global_df.empty:
                paesi = sorted(global_df['PAESE'].unique()) if 'PAESE' in global_df.columns else []
                w_paese.options = paesi
                if paesi: w_paese.value = paesi[0]
                print(f"✅ File caricato: {len(global_df)} righe.")
            else:
                print("❌ Errore caricamento.")

def update_leghe(*args):
    if not global_df.empty and w_paese.value:
        leghe = sorted(global_df[global_df['PAESE'] == w_paese.value]['LEGA'].unique())
        w_lega.options = leghe
        if leghe: w_lega.value = leghe[0]

def update_squadre(*args):
    if not global_df.empty and w_paese.value and w_lega.value:
        mask = (global_df['PAESE'] == w_paese.value) & (global_df['LEGA'] == w_lega.value)
        teams = sorted(pd.concat([global_df[mask]['CASA'], global_df[mask]['OSPITE']]).unique())
        w_home.options = teams
        w_away.options = teams
        if len(teams) > 1:
            w_home.value = teams[0]
            w_away.value = teams[1]
        elif len(teams) == 1:
             w_home.value = teams[0]

w_file.observe(on_file_change, names='value')
w_paese.observe(update_leghe, names='value')
w_lega.observe(update_squadre, names='value')

if available_files and available_files[0] != 'Nessun file trovato':
    on_file_change({'new': available_files[0], 'type': 'change', 'name': 'value'})

# ==========================================
# 3. ENGINE DI ANALISI
# ==========================================
def run_analysis(b):
    with out:
        clear_output()
        if global_df.empty: return print("❌ Nessun dato caricato.")
        
        sel_p, sel_l = w_paese.value, w_lega.value
        sel_h, sel_a = w_home.value, w_away.value
        
        if not sel_h or not sel_a:
            print("⚠️ Seleziona le squadre.")
            return

        print(f"⚙️ ELABORAZIONE: {sel_h} vs {sel_a} ({sel_p} - {sel_l})...\n")
        
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

        # Accumulatori
        goals_h = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
        goals_a = {'FT': 0, 'HT': 0, 'S_FT': 0, 'S_HT': 0}
        match_h, match_a = 0, 0
        times_h, times_a, times_league = [], [], []
        
        # Stats Heatmap
        stats_match = {
            sel_h: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
            sel_a: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
        }

        for _, row in df_league.iterrows():
            h, a = row['CASA'], row['OSPITE']
            min_h = get_minutes(row.get(c_h))
            min_a = get_minutes(row.get(c_a))
            
            # Dati Lega (per Media)
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

            # Dati Match
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

        # Medie Sicure (Diviso 0 check)
        def safe_div(n, d): return n / d if d > 0 else 0

        avg_h_ft = safe_div(goals_h['FT'], match_h)
        avg_h_ht = safe_div(goals_h['HT'], match_h)
        avg_h_conc_ft = safe_div
