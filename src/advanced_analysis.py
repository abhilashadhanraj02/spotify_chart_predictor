"""
spotify_analysis/src/advanced_analysis.py
───────────────────────────────────────────
Three advanced analyses:
  1. FOLLOWER THRESHOLD BREAKPOINT  — exact inflection point
  2. TIME TREND ANALYSIS (1921–2020) — how the winning formula evolved
  3. FEATURE INTERACTION EFFECTS    — synergy between audio features
"""

import ast, json, logging, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_THRESHOLD = 70

TREND_FEATURES = ["danceability","energy","valence","loudness_norm",
                   "acousticness","duration_min","tempo_norm",
                   "speechiness","instrumentalness"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def wilson_ci(s, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = s/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    m = (z*np.sqrt(p*(1-p)/n+z**2/(4*n**2)))/d
    return (round(max(0,c-m),4), round(min(1,c+m),4))

def cohen_h(p1,p2):
    return round(abs(2*np.arcsin(np.sqrt(max(0,p1)))-2*np.arcsin(np.sqrt(max(0,p2)))),4)

def ab_test(ga,gb,la,lb):
    na,nb=len(ga),len(gb); sa,sb=int(ga.sum()),int(gb.sum())
    ra,rb=sa/na if na else 0,sb/nb if nb else 0
    ct=np.array([[max(sa,1),max(na-sa,1)],[max(sb,1),max(nb-sb,1)]])
    _,p,_,_=chi2_contingency(ct,correction=False); h=cohen_h(ra,rb)
    eff="negligible" if h<0.2 else "small" if h<0.5 else "medium" if h<0.8 else "large"
    return {"group_a":la,"group_b":lb,"n_a":na,"n_b":nb,
            "chart_rate_a":round(ra,4),"chart_rate_b":round(rb,4),
            "ci_a":wilson_ci(sa,na),"ci_b":wilson_ci(sb,nb),
            "p_value":round(float(p),6),"significant":bool(p<0.05),
            "cohen_h":h,"effect_size":eff,"winner":la if ra>rb else lb,
            "lift":round((ra-rb)/rb,4) if rb>0 else None}

def print_ab(r):
    sig="✓ SIGNIFICANT" if r["significant"] else "✗ not significant"
    print(f"\n  {r['group_a']}  vs  {r['group_b']}")
    print(f"  {'Group':<38} {'N':>8} {'Chart%':>8}  95% CI")
    print(f"  {'-'*72}")
    print(f"  {r['group_a']:<38} {r['n_a']:>8,} {r['chart_rate_a']:>7.2%}  [{r['ci_a'][0]:.3f},{r['ci_a'][1]:.3f}]")
    print(f"  {r['group_b']:<38} {r['n_b']:>8,} {r['chart_rate_b']:>7.2%}  [{r['ci_b'][0]:.3f},{r['ci_b'][1]:.3f}]")
    print(f"  p={r['p_value']:.4f}  h={r['cohen_h']:.3f} ({r['effect_size']})  {sig}")
    if r["lift"]: print(f"  Lift: {r['lift']:+.1%}  Winner: ★ {r['winner']}")

def section(t):
    print("\n"+"═"*62+f"\n  {t}\n"+"═"*62)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    tracks = pd.read_csv(DATA_DIR/"spotify_tracks.csv", low_memory=False)
    tracks = tracks.rename(columns={"id":"track_id","name":"track_name",
                                     "artists":"artist_names","id_artists":"artist_ids"})
    if "charted" not in tracks.columns:
        tracks["charted"] = (tracks["popularity"] >= CHART_THRESHOLD).astype(int)
    if "loudness_norm" not in tracks.columns and "loudness" in tracks.columns:
        tracks["loudness_norm"] = ((tracks["loudness"]-(-60))/60).clip(0,1)
    if "tempo_norm" not in tracks.columns and "tempo" in tracks.columns:
        tracks["tempo_norm"] = ((tracks["tempo"]-40)/180).clip(0,1)
    if "duration_min" not in tracks.columns and "duration_ms" in tracks.columns:
        tracks["duration_min"] = tracks["duration_ms"]/60_000
    if "energy_dance" not in tracks.columns:
        tracks["energy_dance"] = tracks["energy"]*tracks["danceability"]
    if "release_year" not in tracks.columns and "release_date" in tracks.columns:
        tracks["release_year"] = pd.to_datetime(
            tracks["release_date"],errors="coerce").dt.year

    artists = pd.read_csv(DATA_DIR/"artists.csv", low_memory=False)
    artists = artists.rename(columns={"id":"artist_id","name":"artist_name"})
    artists["followers"] = pd.to_numeric(artists["followers"],errors="coerce").fillna(0)

    def first_id(s):
        try:
            ids=ast.literal_eval(str(s)); return ids[0] if ids else None
        except:
            m=re.search(r"'([^']+)'",str(s)); return m.group(1) if m else None

    tracks["primary_artist_id"] = tracks["artist_ids"].apply(first_id)
    df = tracks.merge(artists[["artist_id","followers"]],
                      left_on="primary_artist_id",right_on="artist_id",how="left")
    df["followers"] = df["followers"].fillna(0)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — FOLLOWER THRESHOLD BREAKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def follower_threshold_analysis(df):
    section("ANALYSIS 1 — FOLLOWER THRESHOLD BREAKPOINT")
    print("  Finding the exact follower count where chart probability inflects.")

    df = df[df["followers"]>0].copy()
    df["log_followers"] = np.log10(df["followers"].clip(1))

    # Fine-grained chart rate by log-follower bucket
    df["log_bucket"] = pd.cut(df["log_followers"], bins=np.arange(0,8.5,0.5))
    bstats = (df.groupby("log_bucket",observed=True)
               .agg(n=("charted","count"),charted=("charted","sum"))
               .reset_index())
    bstats["chart_rate"] = bstats["charted"]/bstats["n"]
    bstats = bstats[bstats["n"]>=100]
    bstats["mid"] = bstats["log_bucket"].apply(lambda x:(x.left+x.right)/2)

    print(f"\n  Chart rate by follower count (log10 buckets):")
    print(f"  {'Followers range':<22} {'N':>8} {'Chart%':>8}  Bar")
    print(f"  {'-'*60}")
    buckets_out = []
    for _,row in bstats.iterrows():
        lo=10**row["log_bucket"].left; hi=10**row["log_bucket"].right
        label=f"{lo:>8,.0f} – {hi:<10,.0f}"
        bar="█"*int(row["chart_rate"]*500)
        print(f"  {label} {row['n']:>8,} {row['chart_rate']:>7.2%}  {bar}")
        buckets_out.append({"followers_approx":round(float(10**row["mid"])),
                            "n":int(row["n"]),"chart_rate":round(float(row["chart_rate"]),4)})

    # Logistic regression
    sample = df.sample(min(100_000,len(df)),random_state=42)
    X = sample[["log_followers"]].values; y = sample["charted"].values
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=500); lr.fit(Xs,y)
    auc = roc_auc_score(y, lr.predict_proba(Xs)[:,1])
    coef = float(lr.coef_[0][0])
    print(f"\n  Logistic regression: log_followers → P(chart)")
    print(f"  Coefficient: {coef:+.4f}  (positive = more followers = more charting)")
    print(f"  AUC-ROC    : {auc:.4f}")

    # Breakpoint — biggest jump between consecutive buckets
    rates = bstats["chart_rate"].values
    mids  = bstats["mid"].values
    jumps = np.diff(rates)
    bp_followers = bp_log = bp_jump = None
    if len(jumps)>0:
        bi = int(np.argmax(jumps))
        bp_log = float(mids[bi+1])
        bp_followers = round(10**bp_log)
        bp_jump = float(jumps[bi])
        print(f"\n  Piecewise breakpoint:")
        print(f"  ★  Threshold : {bp_followers:>12,.0f} followers")
        print(f"     Log10     : {bp_log:.2f}")
        print(f"     Jump      : +{bp_jump:.3%} chart rate at this point")
        print(f"\n  Below {bp_followers:,.0f} followers → audio features are the lever")
        print(f"  Above {bp_followers:,.0f} followers → fanbase carries the song")

        below = df[df["followers"]< bp_followers]["charted"]
        above = df[df["followers"]>=bp_followers]["charted"]
        r=ab_test(above,below,
                  f"Above threshold (≥{bp_followers:,.0f} followers)",
                  f"Below threshold (<{bp_followers:,.0f} followers)")
        print_ab(r)

    return {"buckets":buckets_out,"logistic_auc":round(auc,4),
            "logistic_coef":round(coef,4),"breakpoint_followers":bp_followers,
            "breakpoint_jump":round(bp_jump,4) if bp_jump else None}

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — TIME TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def time_trend_analysis(df):
    section("ANALYSIS 2 — TIME TREND ANALYSIS (1921–2020)")
    print("  How has the winning formula evolved decade by decade?")

    df = df.dropna(subset=["release_year"]).copy()
    df["decade"] = (df["release_year"]//10*10).astype(int)
    df = df[(df["decade"]>=1920)&(df["decade"]<=2020)]

    decade_means  = df.groupby("decade")[TREND_FEATURES+["charted"]].mean()
    decade_counts = df.groupby("decade").size()

    # Feature evolution table
    show_feats = ["energy","acousticness","danceability","loudness_norm","duration_min"]
    print(f"\n  Feature evolution by decade:")
    print(f"  {'Decade':<8} {'N':>7}" + "".join(f"{f[:9]:>11}" for f in show_feats))
    print(f"  {'-'*70}")
    decade_results = []
    for decade,row in decade_means.iterrows():
        n = decade_counts.get(decade,0)
        if n < 50: continue
        vals = "".join(f"{row[f]:>11.3f}" for f in show_feats if f in row.index)
        print(f"  {decade:<8} {n:>7,}{vals}")
        decade_results.append({"decade":int(decade),"n":int(n),
            "chart_rate":round(float(row["charted"]),4),
            **{f:round(float(row[f]),4) for f in TREND_FEATURES if f in row.index}})

    # Linear trend per feature
    print(f"\n  Linear trend — slope per decade, p-value:")
    print(f"  {'Feature':<22} {'Slope/decade':>14} {'p':>10}  Direction")
    print(f"  {'-'*58}")
    valid_decades = [d for d in decade_means.index if decade_counts.get(d,0)>=50]
    trend_results = {}
    for feat in TREND_FEATURES:
        if feat not in decade_means.columns: continue
        y_vals = decade_means.loc[valid_decades,feat].values
        if len(y_vals)<3: continue
        slope,_,_,p,_ = stats.linregress(valid_decades,y_vals)
        direction = "↑ increasing" if slope>0 else "↓ decreasing"
        sig = "★★★" if p<0.001 else ("★★" if p<0.01 else ("★" if p<0.05 else ""))
        print(f"  {feat:<22} {slope:>+14.5f} {p:>10.4f}  {direction} {sig}")
        trend_results[feat] = {"slope_per_decade":round(float(slope),6),
                               "p_value":round(float(p),6),"direction":direction,
                               "significant":bool(p<0.05)}

    # Energy gap widening
    print(f"\n  Is the energy gap between charted/non-charted widening over time?")
    print(f"  {'Decade':<8} {'Charted μ':>11} {'Non-chart μ':>12} {'Gap':>8}")
    print(f"  {'-'*44}")
    gap_results = []
    for decade,ddf in df.groupby("decade"):
        if len(ddf)<50: continue
        c_mean  = ddf[ddf["charted"]==1]["energy"].mean()
        nc_mean = ddf[ddf["charted"]==0]["energy"].mean()
        if np.isnan(c_mean) or np.isnan(nc_mean): continue
        gap = c_mean-nc_mean
        print(f"  {decade:<8} {c_mean:>11.4f} {nc_mean:>12.4f} {gap:>+8.4f}")
        gap_results.append({"decade":int(decade),"charted_energy":round(float(c_mean),4),
                            "noncharted_energy":round(float(nc_mean),4),"gap":round(float(gap),4)})

    # Narrative shifts
    if len(decade_results)>=2:
        first,last = decade_results[0],decade_results[-1]
        print(f"\n  Overall shifts {first['decade']}s → {last['decade']}s:")
        for feat in ["energy","acousticness","loudness_norm","duration_min","danceability"]:
            if feat in first and feat in last:
                delta=last[feat]-first[feat]
                arrow="↑" if delta>0 else "↓"
                print(f"  {arrow} {feat:<22} {first[feat]:.3f} → {last[feat]:.3f}  ({delta:+.3f})")

    # Decade with biggest chart rate
    best_dec = max(decade_results,key=lambda x:x["chart_rate"])
    print(f"\n  ★  Highest chart rate decade: {best_dec['decade']}s "
          f"({best_dec['chart_rate']:.2%})")

    return {"decades":decade_results,"feature_trends":trend_results,"energy_gap":gap_results}

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — FEATURE INTERACTION EFFECTS
# ══════════════════════════════════════════════════════════════════════════════

def feature_interaction_analysis(df):
    section("ANALYSIS 3 — FEATURE INTERACTION EFFECTS")
    print("  Do feature combinations predict charting better than features alone?")

    results = {}

    # Individual vs interaction correlations
    print(f"\n  Correlation with charted — individual vs interaction term:")
    print(f"  {'Predictor':<30} {'|r|':>8}  vs individual")
    print(f"  {'-'*52}")

    pairs = [("energy","danceability"),("energy","loudness_norm"),
             ("danceability","loudness_norm"),("danceability","valence"),
             ("energy","valence")]
    interaction_results = []
    for fa,fb in pairs:
        if fa not in df.columns or fb not in df.columns: continue
        ra,_ = pearsonr(df[fa].fillna(0),df["charted"])
        rb,_ = pearsonr(df[fb].fillna(0),df["charted"])
        ri,_ = pearsonr((df[fa]*df[fb]).fillna(0),df["charted"])
        best_ind = max(abs(ra),abs(rb))
        improvement = abs(ri)-best_ind
        marker = "★ SYNERGY" if improvement>0.002 else ""
        print(f"  {fa:<15} alone     {abs(ra):>8.4f}")
        print(f"  {fb:<15} alone     {abs(rb):>8.4f}")
        print(f"  {fa[:7]}×{fb:<15} {abs(ri):>8.4f}  {improvement:>+.4f}  {marker}")
        print()
        interaction_results.append({"feat_a":fa,"feat_b":fb,
            "r_a":round(float(ra),4),"r_b":round(float(rb),4),
            "r_interaction":round(float(ri),4),
            "improvement":round(float(improvement),4),"synergy":bool(improvement>0.002)})
    results["interactions"] = interaction_results

    # Quadrant tests
    print(f"  Quadrant A/B tests — joint HIGH vs joint LOW on two features:")
    quad_tests = [
        ("energy","danceability",0.7,0.7),
        ("energy","loudness_norm",0.7,0.7),
        ("danceability","loudness_norm",0.7,0.7),
    ]
    quad_results = []
    for fa,fb,ta,tb in quad_tests:
        if fa not in df.columns or fb not in df.columns: continue
        joint_hi = df[(df[fa]>=ta)&(df[fb]>=tb)]
        joint_lo = df[(df[fa]<=(1-ta))&(df[fb]<=(1-tb))]
        if len(joint_hi)<30 or len(joint_lo)<30: continue
        r=ab_test(joint_hi["charted"],joint_lo["charted"],
                  f"High {fa} + High {fb}",f"Low {fa} + Low {fb}")
        print_ab(r)
        rate_joint=joint_hi["charted"].mean()
        rate_a=df[df[fa]>=ta]["charted"].mean()
        rate_b=df[df[fb]>=tb]["charted"].mean()
        synergy=rate_joint-max(rate_a,rate_b)
        print(f"  Joint rate {rate_joint:.3%} vs best single {max(rate_a,rate_b):.3%} "
              f"→ synergy {synergy:+.3%} "
              f"{'★ POSITIVE' if synergy>0.001 else '(no synergy)'}")
        quad_results.append({"feat_a":fa,"feat_b":fb,
            "joint_rate":round(float(rate_joint),4),
            "best_single_rate":round(float(max(rate_a,rate_b)),4),
            "synergy":round(float(synergy),4),"positive_synergy":bool(synergy>0.001)})
    results["quadrant_tests"] = quad_results

    # SHAP from saved model
    print(f"\n  SHAP feature importance from saved Random Forest:")
    try:
        import joblib, shap
        rf      = joblib.load(MODELS_DIR/"random_forest.joblib")
        imputer = joblib.load(MODELS_DIR/"imputer.joblib")
        meta    = json.loads((MODELS_DIR/"results.json").read_text())
        feat_cols = meta["feature_cols"]
        sample = df.sample(min(300,len(df)),random_state=42)
        X = imputer.transform(sample.reindex(columns=feat_cols,fill_value=0))
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X)
        if isinstance(sv,list): sv=np.array(sv[1])
        else:
            sv=np.array(sv)
            if sv.ndim==3: sv=sv[:,:,1]
        mean_abs = np.abs(sv).mean(axis=0).tolist()
        top10 = sorted(zip(feat_cols,mean_abs),key=lambda x:float(x[1]),reverse=True)[:10]
        print(f"  {'Rank':<6} {'Feature':<25} {'Mean |SHAP|':>12}")
        print(f"  {'-'*46}")
        shap_out = []
        for i,(feat,val) in enumerate(top10):
            bar="█"*int(float(val)/float(top10[0][1])*20)
            print(f"  #{i+1:<5} {feat:<25} {float(val):>12.5f}  {bar}")
            shap_out.append({"rank":i+1,"feature":feat,"mean_abs_shap":round(float(val),6)})
        # Check if interaction terms appear
        eng = [f for f,_ in top10 if any(x in f for x in ["energy_dance","loud_energy"])]
        if eng:
            print(f"\n  ★ Engineered interaction features in top 10: {eng}")
        else:
            print(f"\n  Engineered features not in top 10 — interactions captured differently.")
        results["shap_top10"] = shap_out
    except Exception as e:
        print(f"  (Skipped — {e})")
        results["shap_top10"] = []

    # Composite hit score
    print(f"\n  Composite Hit Score = 0.30×energy + 0.25×danceability")
    print(f"                       + 0.25×loudness_norm - 0.20×acousticness")
    df2 = df.copy()
    df2["hit_score"] = (0.30*df2["energy"].fillna(0)
                       +0.25*df2["danceability"].fillna(0)
                       +0.25*df2["loudness_norm"].fillna(0)
                       -0.20*df2["acousticness"].fillna(0))
    buckets = pd.cut(df2["hit_score"],
                     bins=[-9,.3,.45,.55,.65,9],
                     labels=["Very Low","Low","Medium","High","Very High"])
    agg = df2.groupby(buckets,observed=True).agg(
        n=("charted","count"),charted=("charted","sum")).reset_index()
    agg["chart_rate"]=agg["charted"]/agg["n"]
    print(f"\n  {'Score tier':<12} {'N':>8} {'Chart%':>8}  Bar")
    print(f"  {'-'*50}")
    score_results=[]
    for _,row in agg.iterrows():
        bar="█"*int(row["chart_rate"]*500)
        print(f"  {str(row['hit_score']):<12} {row['n']:>8,} {row['chart_rate']:>7.2%}  {bar}")
        score_results.append({"tier":str(row["hit_score"]),"n":int(row["n"]),
                               "chart_rate":round(float(row["chart_rate"]),4)})
    r_score,p_score=pearsonr(df2["hit_score"],df2["charted"])
    print(f"\n  Hit Score ↔ charted: r={r_score:+.4f}  p={p_score:.2e}")
    # best individual was loudness_norm at 0.3271
    best_ind_r = 0.3271
    if abs(r_score)>best_ind_r:
        print(f"  ★ Composite score BEATS best individual feature (loudness_norm r=0.327)!")
    else:
        print(f"  Composite score r={abs(r_score):.4f} vs loudness_norm r={best_ind_r:.4f}")
        print(f"  → Linear combination adds marginal value; non-linear model captures more.")

    vh=df2[df2["hit_score"]>=0.65]["charted"]
    vl=df2[df2["hit_score"]<0.30]["charted"]
    if len(vh)>30 and len(vl)>30:
        r2=ab_test(vh,vl,"Very High Hit Score (≥0.65)","Very Low Hit Score (<0.30)")
        print_ab(r2)

    results["hit_score"]={"buckets":score_results,
                           "correlation":round(float(r_score),4),
                           "beats_best_individual":bool(abs(r_score)>best_ind_r)}
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def run_advanced_analysis(save=True):
    log.info("Loading data…")
    df = load_data()
    log.info("Dataset: %d rows", len(df))
    all_results = {}
    all_results["follower_threshold"]   = follower_threshold_analysis(df)
    all_results["time_trends"]          = time_trend_analysis(df)
    all_results["feature_interactions"] = feature_interaction_analysis(df)
    if save:
        out = OUTPUT_DIR/"advanced_analysis_results.json"
        out.write_text(json.dumps(all_results,indent=2,default=str))
        log.info("Saved → %s", out)
    section("ADVANCED ANALYSIS COMPLETE ✓")
    print(f"  Results → {OUTPUT_DIR/'advanced_analysis_results.json'}")
    return all_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run_advanced_analysis(save=True)
