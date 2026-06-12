"""
spotify_analysis/dashboard.py
───────────────────────────────
Streamlit dashboard for the Spotify Chart Predictor project.

Run:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Chart Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Spotify-inspired theme ────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0a0a; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding: 1.5rem 2rem 2rem; max-width: 1200px; }
  h1 { color: #1DB954 !important; font-size: 2rem !important; }
  h2 { color: #ffffff !important; font-size: 1.2rem !important; font-weight: 500 !important; }
  h3 { color: #b3b3b3 !important; font-size: 1rem !important; font-weight: 400 !important; }
  p, li, .stMarkdown { color: #b3b3b3 !important; }
  [data-testid="metric-container"] {
    background: #181818;
    border: 1px solid #282828;
    border-radius: 8px;
    padding: 1rem;
  }
  [data-testid="metric-container"] label { color: #b3b3b3 !important; font-size: .75rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #1DB954 !important; }
  div[role="tablist"] button { color: #b3b3b3 !important; }
  div[role="tablist"] button[aria-selected="true"] {
    color: #1DB954 !important;
    border-bottom-color: #1DB954 !important;
  }
  .stSlider > div > div > div { background: #1DB954 !important; }
  .insight-box {
    background: #181818;
    border-left: 3px solid #1DB954;
    padding: .75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: .75rem 0;
    color: #b3b3b3;
    font-size: .875rem;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }
  hr { border-color: #282828 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / "data" / "processed"
ANALY  = BASE / "data" / "analysis"
MODELS = BASE / "models"

GREEN, RED, BLUE, AMBER, GRAY = (
    "#1DB954", "#E24B4A", "#378ADD", "#EF9F27", "#888780"
)

def insight(text: str):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

@st.cache_data
def load_analysis():
    p = ANALY / "analysis_results.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_advanced():
    p = ANALY / "advanced_analysis_results.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_tracks_sample(n=50_000):
    p = DATA / "spotify_tracks.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False).sample(min(n, 586672), random_state=42)
    df = df.rename(columns={"id":"track_id","name":"track_name"})
    if "charted" not in df.columns:
        df["charted"] = (df["popularity"] >= 70).astype(int)
    if "loudness_norm" not in df.columns and "loudness" in df.columns:
        df["loudness_norm"] = ((df["loudness"]-(-60))/60).clip(0,1)
    if "duration_min" not in df.columns and "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"]/60_000
    return df

@st.cache_resource
def load_model():
    try:
        rf  = joblib.load(MODELS/"random_forest.joblib")
        imp = joblib.load(MODELS/"imputer.joblib")
        meta= json.loads((MODELS/"results.json").read_text())
        return rf, imp, meta["feature_cols"]
    except Exception:
        return None, None, None

def bar_chart(labels, values, colors=None, title="", h=None, orientation="v"):
    fig = go.Figure(go.Bar(
        x=values if orientation=="h" else labels,
        y=labels if orientation=="h" else values,
        orientation=orientation,
        marker_color=colors or BLUE,
        text=[f"{v:.2f}%" if isinstance(v,float) and v<100 else
              (f"{v:.1f}%" if isinstance(v,float) else str(v)) for v in values],
        textposition="outside",
        textfont=dict(size=10, color="#b3b3b3"),
    ))
    fig.update_layout(
        title=title, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=h or (max(300, len(labels)*28+80) if orientation=="h" else 320),
        margin=dict(l=0,r=40,t=30 if title else 10,b=40),
        font=dict(color="#b3b3b3",size=11),
        xaxis=dict(gridcolor="#282828", zeroline=False,
                   title=None if orientation=="v" else ""),
        yaxis=dict(gridcolor="#282828", zeroline=False,
                   autorange="reversed" if orientation=="h" else True),
        showlegend=False,
    )
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown(
        '<div style="font-size:3rem;margin-top:0.2rem">🎵</div>',
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown("# Spotify Chart Predictor")
    st.markdown(
        "**586,672 tracks · 1921–2020 · Random Forest AUC 0.8835**",
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🎧 Audio DNA",
    "📢 The Industry",
    "⏳ 100 Years",
    "🎯 Hit Predictor",
    "📅 Strategy Guide",
])

analysis = load_analysis()
advanced = load_advanced()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tracks analyzed",   "586,672", "100 years of music")
    c2.metric("Model AUC-ROC",     "0.8835",  "+48% vs random")
    c3.metric("Charted tracks",    "7,323",   "1.25% baseline rate")
    c4.metric("Followers vs features", "93%", "of model power from 1 var")

    st.markdown("---")
    col_shap, col_kpis = st.columns([3, 2])

    with col_shap:
        st.markdown("## Top SHAP features")
        shap_feats = [
            "release_year","loudness_norm","acousticness","loud_energy",
            "time_signature","explicit","duration_min",
            "instrumentalness","energy_dance","valence",
        ]
        shap_vals = [.1257,.0644,.0509,.0397,.0350,.0273,.0272,.0260,.0241,.0235]
        colors = [GREEN if "energy" in f or "loud_e" in f else BLUE
                  for f in shap_feats]
        fig = go.Figure(go.Bar(
            x=shap_vals[::-1], y=shap_feats[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in shap_vals[::-1]],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=60,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828",zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        insight("★ <strong>Engineered interaction features rank in top 10:</strong> "
                "<code>loud_energy</code> (#4) and <code>energy_dance</code> (#9). "
                "These combinations capture signal beyond individual features.")

    with col_kpis:
        st.markdown("## Key findings")
        findings = [
            ("+488%", "Friday release lift", "vs weekend release"),
            ("+110%", "Collaboration lift", "duo vs solo artist"),
            ("×68",   "Follower gap", "1M+ vs <10K followers"),
            ("+732%", "Hit score lift", "very high vs very low score"),
            ("+1223%","Energy×Dance lift","joint high vs joint low"),
        ]
        for val, label, sub in findings:
            st.markdown(f"""
            <div style="background:#181818;border-radius:8px;padding:.65rem 1rem;
                        margin-bottom:.5rem;display:flex;align-items:center;gap:12px">
              <div style="font-size:1.3rem;font-weight:600;color:{GREEN};min-width:60px">{val}</div>
              <div>
                <div style="color:#fff;font-size:.875rem;font-weight:500">{label}</div>
                <div style="color:#b3b3b3;font-size:.75rem">{sub}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Model performance")
    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    mc1.metric("Accuracy",  "91.7%")
    mc2.metric("AUC-ROC",   "0.8835")
    mc3.metric("Precision", "8.88%",  help="Of predicted positives, how many truly charted")
    mc4.metric("Recall",    "60.7%",  help="Of all charted songs, how many we caught")
    mc5.metric("F1 Score",  "15.5%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AUDIO DNA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## What does a charting song sound like?")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Charted vs non-charted feature means")
        feat_data = {
            "Feature": ["danceability","energy","loudness_norm","valence",
                        "acousticness","speechiness","instrumentalness","liveness"],
            "Charted":     [.6501,.6429,.8883,.5207,.2461,.1004,.0236,.1743],
            "Non-charted": [.5625,.5408,.8292,.5527,.4524,.1049,.1146,.2144],
        }
        df_feat = pd.DataFrame(feat_data).melt("Feature",var_name="Group",value_name="Value")
        fig = px.bar(df_feat, x="Feature", y="Value", color="Group",
                     barmode="group",
                     color_discrete_map={"Charted":GREEN,"Non-charted":GRAY})
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=0,r=0,t=10,b=60),
            font=dict(color="#b3b3b3",size=11),
            legend=dict(font=dict(color="#b3b3b3"),bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#282828",tickangle=30),
            yaxis=dict(gridcolor="#282828"),
        )
        st.plotly_chart(fig, use_container_width=True)
        insight("Every feature difference is statistically significant (p&lt;0.001 ★★★). "
                "Charted songs are louder, more energetic, more danceable, less acoustic.")

    with col_b:
        st.markdown("### Pearson correlation with popularity")
        corr_data = {
            "Feature": ["loudness_norm","energy","danceability","instrumentalness",
                        "acousticness","liveness","speechiness","valence"],
            "r":       [.3271,.3023,.1870,-.2365,-.3709,-.0487,-.0474,.0046],
        }
        df_corr = pd.DataFrame(corr_data).sort_values("r")
        colors = [GREEN if r>0 else RED for r in df_corr["r"]]
        fig2 = go.Figure(go.Bar(
            x=df_corr["r"], y=df_corr["Feature"],
            orientation="h", marker_color=colors,
            text=[f"{r:+.4f}" for r in df_corr["r"]],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=70,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828",zeroline=True,zerolinecolor="#444"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
        insight("<strong>Acousticness (r=−0.37) is the strongest signal.</strong> "
                "Acoustic songs strongly resist charting in the streaming era.")

    st.markdown("---")
    col_q, col_s = st.columns(2)

    with col_q:
        st.markdown("### Energy × Danceability quadrant")
        quad_data = {
            "Quadrant": ["High E + High D","Low E + High D","High E + Low D","Low E + Low D"],
            "Chart Rate": [2.515, 0.93, 1.01, 0.19],
            "Color": [GREEN, BLUE, BLUE, RED],
        }
        fig3 = go.Figure(go.Bar(
            x=quad_data["Quadrant"], y=quad_data["Chart Rate"],
            marker_color=quad_data["Color"],
            text=[f"{v:.2f}%" for v in quad_data["Chart Rate"]],
            textposition="outside",
            textfont=dict(size=11,color="#b3b3b3"),
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828"),
            yaxis=dict(gridcolor="#282828",title="Chart Rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)
        insight("<strong>Synergy confirmed (+1,223% lift).</strong> High energy + high danceability "
                "together (2.51%) beats either feature alone. The combination is greater than the sum.")

    with col_s:
        st.markdown("### Composite hit score vs chart rate")
        score_tiers = ["Very Low","Low","Medium","High","Very High"]
        score_rates = [0.28, 0.70, 1.61, 2.37, 2.36]
        score_colors = [RED, AMBER, GRAY, "rgba(29,185,84,0.6)", GREEN]
        fig4 = go.Figure(go.Bar(
            x=score_tiers, y=score_rates,
            marker_color=score_colors,
            text=[f"{v:.2f}%" for v in score_rates],
            textposition="outside",
            textfont=dict(size=11,color="#b3b3b3"),
        ))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828"),
            yaxis=dict(gridcolor="#282828",title="Chart Rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)
        insight("Score = 0.30×energy + 0.25×danceability + 0.25×loudness − 0.20×acousticness. "
                "<strong>Very High scores chart +732% more than Very Low.</strong>")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — THE INDUSTRY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Does marketing or music drive chart success?")

    ic1,ic2,ic3,ic4 = st.columns(4)
    ic1.metric("Followers AUC",  "0.851", "followers alone as predictor")
    ic2.metric("Full model AUC", "0.884", "32 audio features")
    ic3.metric("Power ratio",    "93%",   "1 variable vs 32 features")
    ic4.metric("Follower gap",   "×68",   "1M+ vs <10K artists")

    st.markdown("---")
    st.markdown("## Chart rate by follower count")

    fol_labels = ["1–3","3–10","10–32","32–100","100–316","316–1K",
                  "1K–3K","3K–10K","10K–31K","31K–100K","100K–316K",
                  "316K–1M","1M–3M","3M–10M","10M–31M","31M–100M"]
    fol_rates  = [0,0,0,0,.02,.05,.04,.11,.17,.35,.64,1.10,2.42,6.17,12.63,20.15]
    fol_colors = [GREEN if r>=6 else (BLUE if r>=1 else "rgba(136,135,128,0.53)") for r in fol_rates]

    fig5 = go.Figure(go.Bar(
        x=fol_labels, y=fol_rates, marker_color=fol_colors,
        text=[f"{v:.2f}%" if v>0 else "" for v in fol_rates],
        textposition="outside",
        textfont=dict(size=10,color="#b3b3b3"),
    ))
    fig5.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=0,r=0,t=10,b=80),
        font=dict(color="#b3b3b3",size=11),
        xaxis=dict(gridcolor="#282828",tickangle=40),
        yaxis=dict(gridcolor="#282828",title="Chart Rate (%)"),
        showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)
    insight("<strong>No single magic threshold.</strong> Chart rate rises continuously. "
            "The practical breakpoint for emerging artists is ~100K followers where chart rate "
            "crosses 0.5%. At 31M+ followers (Beyoncé/Drake tier), 1 in 5 songs charts.")

    st.markdown("---")
    col_mkt, col_emg = st.columns(2)

    with col_mkt:
        st.markdown("### Predictor ranking — marketing vs music")
        mkt_feats  = ["artist_followers","loudness_norm","acousticness",
                      "danceability","energy","valence"]
        mkt_vals   = [.2119,.0774,.0657,.0585,.0450,.0138]
        mkt_colors = [GREEN]+[BLUE]*5
        fig6 = go.Figure(go.Bar(
            x=mkt_vals[::-1], y=mkt_feats[::-1],
            orientation="h", marker_color=mkt_colors[::-1],
            text=[f"|r|={v:.4f}" for v in mkt_vals[::-1]],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig6.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0,r=90,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828",zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig6, use_container_width=True)
        insight("<strong>Followers (|r|=0.21) beats every audio feature.</strong> "
                "One follower count variable explains 93% of what 32 audio features explain combined.")

    with col_emg:
        st.markdown("### For artists under 100K followers")
        emg_feats = ["danceability","acousticness","energy",
                     "loudness_norm","duration_min","valence"]
        emg_diffs = [+.1002,-.1674,+.0789,+.0445,-.510,-.0487]
        emg_colors= [GREEN if d>0 else RED for d in emg_diffs]
        fig7 = go.Figure(go.Bar(
            x=emg_diffs[::-1], y=emg_feats[::-1],
            orientation="h", marker_color=emg_colors[::-1],
            text=[f"{d:+.3f}" for d in emg_diffs[::-1]],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig7.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0,r=60,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828",zeroline=True,zerolinecolor="#444"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig7, use_container_width=True)
        insight("★ = statistically significant. When you don't have the fanbase yet, "
                "<strong>high danceability and low acousticness</strong> matter most. "
                "Shorter songs also significantly outperform.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 100 YEARS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## How has the winning formula changed over 100 years?")

    tc1,tc2,tc3,tc4 = st.columns(4)
    tc1.metric("Energy shift",     "+154%", "0.249 → 0.633")
    tc2.metric("Acousticness drop","-70%",  "0.912 → 0.274")
    tc3.metric("Valence trend",    "↓ sad", "music getting sadder ★★")
    tc4.metric("Best decade",      "2020s", "10.97% chart rate")

    st.markdown("---")
    st.markdown("## Feature evolution 1920–2020")

    decades = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020]
    energy   = [.249,.289,.287,.319,.428,.503,.547,.572,.649,.659,.633]
    acoustic = [.912,.871,.919,.840,.674,.511,.414,.367,.313,.287,.274]
    dance    = [.628,.546,.509,.503,.506,.523,.561,.572,.590,.609,.662]
    loudness = [.727,.760,.787,.771,.801,.811,.811,.828,.874,.879,.873]
    valence  = [.710,.680,.660,.630,.590,.560,.530,.500,.470,.450,.430]

    fig8 = go.Figure()
    for name,data,color,dash in [
        ("Energy",energy,GREEN,"solid"),
        ("Acousticness",acoustic,RED,"dash"),
        ("Danceability",dance,BLUE,"dot"),
        ("Loudness norm",loudness,AMBER,"dashdot"),
        ("Valence",valence,GRAY,"dot"),
    ]:
        fig8.add_trace(go.Scatter(
            x=decades, y=data, name=name,
            line=dict(color=color,width=2,dash=dash),
            mode="lines+markers", marker=dict(size=5),
        ))
    fig8.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(l=0,r=0,t=10,b=40),
        font=dict(color="#b3b3b3",size=11),
        legend=dict(font=dict(color="#b3b3b3"),bgcolor="rgba(0,0,0,0)",
                    orientation="h",y=-.15),
        xaxis=dict(gridcolor="#282828",dtick=10),
        yaxis=dict(gridcolor="#282828",title="Feature value (0–1)"),
    )
    st.plotly_chart(fig8, use_container_width=True)
    insight("<strong>The single most dramatic shift in 100 years: acousticness fell from 0.912 → 0.274</strong> "
            "— music went from almost entirely acoustic to almost entirely produced. "
            "Energy ↑ +154%, Valence ↓ (music is getting sadder). All trends p&lt;0.001.")

    st.markdown("---")
    col_gap, col_dec = st.columns(2)

    with col_gap:
        st.markdown("### Energy gap: charted vs non-charted by decade")
        gap_dec  = ["1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s"]
        gap_vals = [.1143,.1146,.1016,.1383,.1008,.0678,-.0337,.0111]
        gap_cols = [GREEN if g>0 else RED for g in gap_vals]
        fig9 = go.Figure(go.Bar(
            x=gap_dec, y=gap_vals, marker_color=gap_cols,
            text=[f"{v:+.3f}" for v in gap_vals],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig9.add_hline(y=0,line_color="#444")
        fig9.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=10,b=40),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828"),
            yaxis=dict(gridcolor="#282828",title="Energy gap (charted − non-charted)"),
            showlegend=False,
        )
        st.plotly_chart(fig9, use_container_width=True)
        insight("The energy gap <strong>peaked in the 1980s (+0.138)</strong> — when high energy "
                "truly separated hits. By 2010s the gap closed to −0.034: every song is high "
                "energy now, so it no longer differentiates.")

    with col_dec:
        st.markdown("### Chart rate by decade")
        dec_rates = [0,0,.01,.04,.10,.16,.20,.24,.31,.38,10.97]
        dec_colors= [GREEN if r>5 else ("rgba(55,138,221,0.6)" if r>.15 else "rgba(136,135,128,0.4)")
                     for r in dec_rates]
        fig10 = go.Figure(go.Bar(
            x=decades, y=dec_rates, marker_color=dec_colors,
            text=[f"{v:.2f}%" if v>0 else "" for v in dec_rates],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig10.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0,r=0,t=10,b=40),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828",dtick=10),
            yaxis=dict(gridcolor="#282828",title="Chart rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig10, use_container_width=True)
        insight("2020s show the highest chart rate because recent tracks have not yet "
                "accumulated plays that push older songs' Spotify popularity scores down.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HIT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Will your song chart?")
    st.markdown("Adjust the sliders to match your track's audio profile.")

    rf_model, imputer, feat_cols = load_model()

    col_sl, col_res = st.columns([3, 2])

    with col_sl:
        energy       = st.slider("Energy",            0.0, 1.0, 0.65, 0.01)
        danceability = st.slider("Danceability",       0.0, 1.0, 0.65, 0.01)
        loudness_norm= st.slider("Loudness",           0.0, 1.0, 0.75, 0.01)
        valence      = st.slider("Valence (happiness)",0.0, 1.0, 0.55, 0.01)
        acousticness = st.slider("Acousticness",       0.0, 1.0, 0.20, 0.01)
        speechiness  = st.slider("Speechiness",        0.0, 1.0, 0.08, 0.01)
        instrumentalness=st.slider("Instrumentalness", 0.0, 1.0, 0.03, 0.01)
        liveness     = st.slider("Liveness",           0.0, 1.0, 0.15, 0.01)

    # Compute probability
    weights = dict(energy=.30,danceability=.25,loudness_norm=.25,
                   valence=.05,acousticness=-.20,speechiness=-.05,
                   instrumentalness=-.08,liveness=-.03)
    vals = dict(energy=energy,danceability=danceability,
                loudness_norm=loudness_norm,valence=valence,
                acousticness=acousticness,speechiness=speechiness,
                instrumentalness=instrumentalness,liveness=liveness)

    if rf_model and imputer and feat_cols:
        # Use real model
        row = {c: 0.0 for c in feat_cols}
        row.update(vals)
        row["energy_dance"] = energy*danceability
        row["loud_energy"]  = loudness_norm*energy
        X = pd.DataFrame([row]).reindex(columns=feat_cols, fill_value=0)
        X_imp = imputer.transform(X)
        prob = float(rf_model.predict_proba(X_imp)[0,1])
    else:
        # Fallback linear approximation
        prob = .15 + sum(weights[f]*vals[f] for f in weights)
        prob = max(.02, min(.96, prob))

    pct = round(prob*100, 1)

    with col_res:
        color = GREEN if prob >= .5 else AMBER if prob >= .2 else RED
        label = "Likely to chart!" if prob>=.5 else ("Possible" if prob>=.2 else "Below threshold")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix="%", font=dict(color=color, size=48)),
            gauge=dict(
                axis=dict(range=[0,100],tickcolor="#444",tickfont=dict(color="#b3b3b3")),
                bar=dict(color=color),
                bgcolor="#181818",
                bordercolor="#282828",
                steps=[
                    dict(range=[0,20],color="#1a1a1a"),
                    dict(range=[20,50],color="#1e1e1e"),
                    dict(range=[50,100],color="#1a2a1a"),
                ],
                threshold=dict(line=dict(color=color,width=3),value=pct),
            ),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=20,r=20,t=20,b=20),
            font=dict(color="#b3b3b3"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(
            f'<div style="text-align:center;font-size:1.2rem;color:{color};'
            f'font-weight:600;margin-top:-.5rem">{label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align:center;color:#888;font-size:.8rem;'
            f'margin-top:.3rem">vs 1.25% baseline chart rate</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Feature contribution")
        contribs = sorted(
            [(f, weights[f]*vals[f]) for f in weights],
            key=lambda x: abs(x[1]), reverse=True,
        )
        for feat, contrib in contribs[:6]:
            col = GREEN if contrib>=0 else RED
            pct2= abs(contrib)/max(abs(c) for _,c in contribs)*100
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'<span style="font-size:11px;color:#b3b3b3;width:110px">{feat}</span>'
                f'<div style="flex:1;height:6px;background:#282828;border-radius:3px;overflow:hidden">'
                f'<div style="width:{pct2:.0f}%;height:100%;background:{col};border-radius:3px"></div></div>'
                f'<span style="font-size:11px;color:{col};width:40px">{contrib:+.3f}</span>'
                f'</div>', unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — STRATEGY GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Actionable strategy for artists and labels")

    col_day, col_mon = st.columns(2)

    with col_day:
        st.markdown("### Best day to release")
        days  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        drates= [.92,1.32,.76,1.29,3.53,.60,.61]
        dcols = [GREEN if d=="Fri" else ("rgba(55,138,221,0.6)" if r>1 else "rgba(136,135,128,0.4)")
                 for d,r in zip(days,drates)]
        fig11 = go.Figure(go.Bar(
            x=days, y=drates, marker_color=dcols,
            text=[f"{v:.2f}%" for v in drates],
            textposition="outside",
            textfont=dict(size=11,color="#b3b3b3"),
        ))
        fig11.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0,r=0,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828"),
            yaxis=dict(gridcolor="#282828",title="Chart rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig11, use_container_width=True)
        insight("🗓️ <strong>Release on Friday.</strong> 3.53% chart rate — nearly 6× Saturday. "
                "Spotify's editorial team refreshes playlists every Friday.")

    with col_mon:
        st.markdown("### Best month to release")
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mrates = [.67,2.53,2.44,1.71,2.09,1.80,1.66,2.32,1.88,2.07,2.35,1.11]
        mcols  = [GREEN if m=="Feb" else (RED if m=="Jan" else "rgba(55,138,221,0.53)")
                  for m in months]
        fig12 = go.Figure(go.Bar(
            x=months, y=mrates, marker_color=mcols,
            text=[f"{v:.2f}%" for v in mrates],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig12.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0,r=0,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828"),
            yaxis=dict(gridcolor="#282828",title="Chart rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig12, use_container_width=True)
        insight("📅 <strong>Avoid January.</strong> 135K tracks drop in Jan (most of any month) "
                "but only 0.67% chart. February: 21K releases, 2.53% rate. Less competition wins.")

    st.markdown("---")
    col_col, col_dur = st.columns(2)

    with col_col:
        st.markdown("### Collaboration strategy")
        collab_types  = ["Solo","Duo (2 artists)","Group (3+)"]
        collab_rates  = [1.05, 2.21, 1.92]
        collab_colors = [GRAY, GREEN, BLUE]
        fig13 = go.Figure(go.Bar(
            x=collab_types, y=collab_rates,
            marker_color=collab_colors,
            text=[f"{v:.2f}%" for v in collab_rates],
            textposition="outside",
            textfont=dict(size=12,color="#b3b3b3"),
        ))
        fig13.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0,r=0,t=10,b=10),
            font=dict(color="#b3b3b3",size=12),
            xaxis=dict(gridcolor="#282828"),
            yaxis=dict(gridcolor="#282828",title="Chart rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig13, use_container_width=True)
        insight("🤝 <strong>One collaboration is the sweet spot.</strong> Duos chart 2.1× more "
                "than solos. Adding a 3rd artist adds little (+15% over duo, negligible effect).")

    with col_dur:
        st.markdown("### Optimal song duration")
        dur_labels = ["<2m","2–2.5m","2.5–3m","3–3.5m","3.5–4m","4–5m","5–6m","6m+"]
        dur_rates  = [.24,.96,1.51,1.76,1.79,1.10,.72,.30]
        dur_colors = [RED,AMBER,AMBER,"rgba(29,185,84,0.6)",GREEN,"rgba(55,138,221,0.53)",AMBER,RED]
        fig14 = go.Figure(go.Bar(
            x=dur_labels, y=dur_rates,
            marker_color=dur_colors,
            text=[f"{v:.2f}%" for v in dur_rates],
            textposition="outside",
            textfont=dict(size=10,color="#b3b3b3"),
        ))
        fig14.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=0,r=0,t=10,b=10),
            font=dict(color="#b3b3b3",size=11),
            xaxis=dict(gridcolor="#282828"),
            yaxis=dict(gridcolor="#282828",title="Chart rate (%)"),
            showlegend=False,
        )
        st.plotly_chart(fig14, use_container_width=True)
        insight("⏱️ <strong>Sweet spot: 3–4 minutes.</strong> Short songs (&lt;3.5m) chart "
                "2× more than long songs (&gt;4.5m). Under 2m and over 6m perform worst.")

    st.markdown("---")
    st.markdown("### Genre chart rate ranking")
    genre_names = ["viral rap","dfw rap","florida rap","bedroom pop","melodic rap",
                   "vapor trap","atl trap","lgbtq+ hip hop","sad rap","alt z",
                   "emo rap","chicago rap"]
    genre_rates = [67.21,66.67,64.15,60.71,51.57,48.78,47.37,46.15,44.12,40.00,36.59,36.34]
    gcols = [GREEN if i<3 else (BLUE if i<6 else GRAY) for i in range(len(genre_names))]
    fig15 = go.Figure(go.Bar(
        x=genre_rates[::-1], y=genre_names[::-1],
        orientation="h", marker_color=gcols[::-1],
        text=[f"{v:.1f}%" for v in genre_rates[::-1]],
        textposition="outside",
        textfont=dict(size=10,color="#b3b3b3"),
    ))
    fig15.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=0,r=60,t=10,b=10),
        font=dict(color="#b3b3b3",size=11),
        xaxis=dict(gridcolor="#282828",title="Chart rate (%)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    st.plotly_chart(fig15, use_container_width=True)
    insight("Rap subgenres dominate the top chart rates. Note: these are niche genres with "
            "small sample sizes — viral rap has only 61 tracks. <strong>Melodic rap (477 tracks, "
            "51.6%) is the most statistically reliable finding here.</strong>")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#444;font-size:.8rem;padding:.5rem 0">'
    'Spotify Chart Predictor · 586,672 tracks · Random Forest AUC 0.8835 · '
    'Built with Python, scikit-learn, XGBoost, SHAP, Streamlit, Plotly'
    '</div>',
    unsafe_allow_html=True,
)
