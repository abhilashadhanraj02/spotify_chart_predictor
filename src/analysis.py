"""
spotify_analysis/src/analysis.py
──────────────────────────────────
Comprehensive analysis module covering:
  1.  Audio features vs chart position (danceability, energy, valence, loudness)
  2.  Hit song profile — cluster analysis + top-10% vs non-charting comparison
  3.  Artist popularity vs song characteristics (marketing vs music)
  4.  Low-follower artist success factors (emerging artist analysis)
  5.  Collaboration effect (solo vs 2-artist vs 3+ artists)
  6.  Song duration vs chart success
  7.  Release timing — optimal month and weekday
  8.  A/B tests with chi-squared + Cohen's h effect size
  9.  Gender bias analysis
  10. Genre chart rate ranking
"""

import ast, json, logging, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHART_THRESHOLD = 70
AUDIO_FEATURES  = [
    "danceability","energy","loudness_norm","tempo_norm",
    "speechiness","acousticness","instrumentalness",
    "liveness","valence","duration_min",
]

# ── Statistical helpers ───────────────────────────────────────────────────────

def wilson_ci(s, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = s / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    m = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d
    return (round(max(0,c-m),4), round(min(1,c+m),4))

def cohen_h(p1, p2):
    return round(abs(2*np.arcsin(np.sqrt(max(0,p1))) - 2*np.arcsin(np.sqrt(max(0,p2)))), 4)

def effect_label(h):
    if h < 0.2: return "negligible"
    if h < 0.5: return "small"
    if h < 0.8: return "medium"
    return "large"

def ab_test(ga, gb, la, lb):
    na, nb = len(ga), len(gb)
    sa, sb = int(ga.sum()), int(gb.sum())
    ra, rb = sa/na if na else 0, sb/nb if nb else 0
    ct = np.array([[sa, na-sa],[sb, nb-sb]])
    chi2, p, _, _ = chi2_contingency(ct, correction=False)
    h = cohen_h(ra, rb)
    return {
        "group_a": la, "group_b": lb,
        "n_a": na, "n_b": nb,
        "chart_rate_a": round(ra,4), "chart_rate_b": round(rb,4),
        "ci_a": wilson_ci(sa,na), "ci_b": wilson_ci(sb,nb),
        "p_value": round(float(p),6), "significant": bool(p<0.05),
        "cohen_h": h, "effect_size": effect_label(h),
        "winner": la if ra>rb else lb,
        "lift": round((ra-rb)/rb,4) if rb>0 else None,
    }

def print_ab(r):
    sig = "✓ SIGNIFICANT" if r["significant"] else "✗ not significant"
    print(f"\n  {r['group_a']}  vs  {r['group_b']}")
    print(f"  {'Group':<32} {'N':>8} {'Chart%':>8}  {'95% CI'}")
    print(f"  {'-'*68}")
    print(f"  {r['group_a']:<32} {r['n_a']:>8,} {r['chart_rate_a']:>7.2%}  [{r['ci_a'][0]:.3f},{r['ci_a'][1]:.3f}]")
    print(f"  {r['group_b']:<32} {r['n_b']:>8,} {r['chart_rate_b']:>7.2%}  [{r['ci_b'][0]:.3f},{r['ci_b'][1]:.3f}]")
    print(f"  p={r['p_value']:.4f}  h={r['cohen_h']:.3f} ({r['effect_size']})  {sig}")
    if r["lift"]: print(f"  Lift: {r['lift']:+.1%}  Winner: {r['winner']}")

def section(title):
    print("\n" + "═"*62)
    print(f"  {title}")
    print("═"*62)

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_tracks():
    df = pd.read_csv(DATA_DIR/"spotify_tracks.csv", low_memory=False)
    df = df.rename(columns={"id":"track_id","name":"track_name",
                             "artists":"artist_names","id_artists":"artist_ids"})
    if "charted" not in df.columns:
        df["charted"] = (df["popularity"] >= CHART_THRESHOLD).astype(int)
    if "loudness_norm" not in df.columns and "loudness" in df.columns:
        df["loudness_norm"] = ((df["loudness"]-(-60))/60).clip(0,1)
    if "tempo_norm" not in df.columns and "tempo" in df.columns:
        df["tempo_norm"] = ((df["tempo"]-40)/180).clip(0,1)
    if "duration_min" not in df.columns and "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"]/60000
    if "release_date" in df.columns:
        df["release_dt"]    = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_month"] = df["release_dt"].dt.month
        df["release_dow"]   = df["release_dt"].dt.dayofweek  # 0=Mon
    return df

def load_artists():
    df = pd.read_csv(DATA_DIR/"artists.csv", low_memory=False)
    df = df.rename(columns={"id":"artist_id","name":"artist_name"})
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0)
    return df

def parse_first_artist_id(s):
    try:
        ids = ast.literal_eval(str(s))
        return ids[0] if ids else None
    except:
        m = re.search(r"'([^']+)'", str(s))
        return m.group(1) if m else None

def count_artists(s):
    try:
        ids = ast.literal_eval(str(s))
        return len(ids) if isinstance(ids, list) else 1
    except:
        return 1

def join_artists(tracks, artists):
    tracks = tracks.copy()
    tracks["primary_artist_id"] = tracks["artist_ids"].apply(parse_first_artist_id)
    tracks["n_artists"]         = tracks["artist_ids"].apply(count_artists)
    return tracks.merge(
        artists[["artist_id","followers","genres","popularity"]],
        left_on="primary_artist_id", right_on="artist_id",
        how="left", suffixes=("_track","_artist"),
    ).assign(followers=lambda x: x["followers"].fillna(0))

# ── 1. Audio features vs chart position ──────────────────────────────────────

def audio_vs_chart(df):
    section("1. AUDIO FEATURES vs CHART POSITION")
    results = {}
    feats = ["danceability","energy","valence","loudness_norm","acousticness",
             "speechiness","instrumentalness","liveness","tempo_norm"]
    charted     = df[df["charted"]==1]
    non_charted = df[df["charted"]==0]

    print(f"\n  {'Feature':<22} {'Charted μ':>10} {'Non-chart μ':>12} {'Diff':>8} {'p-val':>10} {'Sig':>5}")
    print(f"  {'-'*70}")

    for feat in feats:
        if feat not in df.columns: continue
        c  = charted[feat].dropna()
        nc = non_charted[feat].dropna()
        _, p = stats.ttest_ind(c, nc, equal_var=False)
        diff = c.mean() - nc.mean()
        sig = "★★★" if p<0.001 else ("★★" if p<0.01 else ("★" if p<0.05 else ""))
        print(f"  {feat:<22} {c.mean():>10.4f} {nc.mean():>12.4f} {diff:>+8.4f} {p:>10.2e} {sig:>5}")
        results[feat] = {
            "charted_mean": round(float(c.mean()),4),
            "noncharted_mean": round(float(nc.mean()),4),
            "diff": round(float(diff),4),
            "p_value": round(float(p),8),
            "significant": bool(p<0.05),
        }

    # Correlation with raw popularity score
    print(f"\n  Pearson correlation with popularity score:")
    corr_results = {}
    for feat in feats:
        if feat not in df.columns: continue
        pop_col = "popularity_track" if "popularity_track" in df.columns else "popularity"
        r, p = stats.pearsonr(df[feat].fillna(0), df[pop_col])
        bar = ("+" if r>0 else "-") + "█"*int(abs(r)*30)
        print(f"  {feat:<22} r={r:>+.4f}  p={p:.2e}  {bar}")
        corr_results[feat] = {"r": round(float(r),4), "p": round(float(p),8)}
    results["correlations"] = corr_results
    return results

# ── 2. Hit song profile — clustering ─────────────────────────────────────────

def hit_song_profile(df):
    section("2. HIT SONG PROFILE — CLUSTER ANALYSIS")
    cluster_feats = ["danceability","energy","valence","loudness_norm",
                     "acousticness","speechiness","instrumentalness",
                     "liveness","tempo_norm"]
    pop_col = "popularity_track" if "popularity_track" in df.columns else "popularity"
    sub = df[cluster_feats + ["charted", pop_col]].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(sub[cluster_feats])

    # K-Means with k=4
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    sub = sub.copy()
    sub["cluster"] = km.fit_predict(X)

    print(f"\n  K-Means clustering (k=4) on {len(sub):,} tracks")
    print(f"\n  {'Cluster':<10} {'N':>8} {'Chart%':>8} {'Avg Pop':>9}  Profile")
    print(f"  {'-'*70}")

    cluster_profiles = []
    for c in sorted(sub["cluster"].unique()):
        cdf  = sub[sub["cluster"]==c]
        rate = cdf["charted"].mean()
        pop_col2 = "popularity_track" if "popularity_track" in cdf.columns else "popularity"
        pop  = cdf[pop_col2].mean()
        # Describe cluster by dominant features
        means = cdf[cluster_feats].mean()
        top3  = means.nlargest(3).index.tolist()
        bot1  = means.nsmallest(1).index.tolist()
        profile = f"High {', '.join(top3)} / Low {bot1[0]}"
        print(f"  Cluster {c:<3} {len(cdf):>8,} {rate:>7.2%} {pop:>9.1f}  {profile}")
        cluster_profiles.append({
            "cluster": int(c), "n": len(cdf),
            "chart_rate": round(float(rate),4),
            "avg_popularity": round(float(pop),2),
            "top_features": top3,
            "feature_means": {f: round(float(means[f]),4) for f in cluster_feats},
        })

    # Top 10% vs non-charting comparison
    pop_col3 = "popularity_track" if "popularity_track" in df.columns else "popularity"
    top10  = df[df[pop_col3] >= df[pop_col3].quantile(0.90)]
    bottom = df[df["charted"] == 0]
    print(f"\n  Top 10% songs vs Non-charting comparison:")
    print(f"  {'Feature':<22} {'Top 10%':>10} {'Non-chart':>10} {'Diff':>8}")
    print(f"  {'-'*54}")
    comparison = {}
    for feat in ["danceability","energy","valence","loudness_norm","acousticness","duration_min"]:
        if feat not in df.columns: continue
        t_mean = top10[feat].mean()
        b_mean = bottom[feat].mean()
        diff   = t_mean - b_mean
        print(f"  {feat:<22} {t_mean:>10.4f} {b_mean:>10.4f} {diff:>+8.4f}")
        comparison[feat] = {"top10_mean": round(float(t_mean),4),
                            "noncharted_mean": round(float(b_mean),4),
                            "diff": round(float(diff),4)}
    return {"clusters": cluster_profiles, "top10_vs_noncharted": comparison}

# ── 3. Artist popularity vs song features ─────────────────────────────────────

def marketing_vs_music(df):
    section("3. MARKETING vs MUSIC — Artist Followers vs Audio Features")

    # Correlation: artist followers vs charted
    fol_corr, fol_p = stats.pointbiserialr(df["followers"].fillna(0), df["charted"])
    print(f"\n  Artist followers ↔ charted:  r={fol_corr:+.4f}  p={fol_p:.2e}")

    # Compare top audio feature correlations vs follower correlation
    print(f"\n  Predictor ranking (correlation with charted label):")
    predictors = {
        "artist_followers": abs(fol_corr),
    }
    feats = ["danceability","energy","valence","loudness_norm","acousticness"]
    for feat in feats:
        if feat in df.columns:
            r, _ = stats.pointbiserialr(df[feat].fillna(0), df["charted"])
            predictors[feat] = abs(r)

    for name, val in sorted(predictors.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(val * 100)
        marker = " ◄ ARTIST" if name == "artist_followers" else ""
        print(f"  {name:<25} |r|={val:.4f}  {bar}{marker}")

    # A/B: high-follower vs low-follower artists
    results = {"follower_vs_features": {k: round(float(v),4) for k,v in predictors.items()}}
    hi = df[df["followers"] >= 1_000_000]["charted"]
    lo = df[df["followers"] <  10_000]["charted"]
    if len(hi)>30 and len(lo)>30:
        r = ab_test(lo, hi, "Low followers (<10K)", "High followers (1M+)")
        print_ab(r)
        results["ab_test"] = r
    return results

# ── 4. Low-follower artist success factors ────────────────────────────────────

def emerging_artist_analysis(df):
    section("4. EMERGING ARTIST ANALYSIS — What makes a low-follower song chart?")

    tiers = {
        "<100K followers":  df[df["followers"] <  100_000],
        "100K–1M":          df[(df["followers"] >= 100_000) & (df["followers"] < 1_000_000)],
        "1M+ followers":    df[df["followers"] >= 1_000_000],
    }

    print(f"\n  {'Tier':<20} {'N':>8} {'Charted':>8} {'Chart%':>8} {'Avg followers':>15}")
    print(f"  {'-'*62}")
    tier_stats = {}
    for name, tdf in tiers.items():
        n = len(tdf); c = tdf["charted"].sum()
        rate = c/n if n else 0
        avg_fol = tdf["followers"].mean()
        print(f"  {name:<20} {n:>8,} {c:>8,} {rate:>7.2%} {avg_fol:>15,.0f}")
        tier_stats[name] = {"n": n, "chart_rate": round(float(rate),4)}

    # A/B tests between tiers
    results = {"tier_stats": tier_stats, "ab_tests": []}
    tier_list = list(tiers.items())
    for i in range(len(tier_list)-1):
        la, ga = tier_list[i]
        lb, gb = tier_list[i+1]
        r = ab_test(ga["charted"], gb["charted"], la, lb)
        print_ab(r)
        results["ab_tests"].append(r)

    # What features separate charted vs non-charted within <100K tier
    small = tiers["<100K followers"]
    sc = small[small["charted"]==1]
    snc= small[small["charted"]==0]
    if len(sc) > 10:
        print(f"\n  Within <100K artists — what separates charted from not?")
        print(f"  (★ = statistically significant, p<0.05)")
        print(f"  {'Feature':<22} {'Charted μ':>10} {'Non-chart μ':>12} {'Diff':>8}")
        print(f"  {'-'*56}")
        feat_diffs = {}
        for feat in ["danceability","energy","valence","loudness_norm",
                     "acousticness","duration_min","tempo_norm"]:
            if feat not in small.columns: continue
            c_m  = sc[feat].mean()
            nc_m = snc[feat].mean()
            _, p = stats.ttest_ind(sc[feat].dropna(), snc[feat].dropna(), equal_var=False)
            sig = "★" if p<0.05 else " "
            print(f"  {sig} {feat:<20} {c_m:>10.4f} {nc_m:>12.4f} {c_m-nc_m:>+8.4f}")
            feat_diffs[feat] = {"charted_mean": round(float(c_m),4),
                                "noncharted_mean": round(float(nc_m),4),
                                "diff": round(float(c_m-nc_m),4),
                                "significant": bool(p<0.05)}
        results["small_artist_feature_diffs"] = feat_diffs
    return results

# ── 5. Collaboration effect ───────────────────────────────────────────────────

def collaboration_analysis(df):
    section("5. COLLABORATION EFFECT")

    df = df.copy()
    df["collab_tier"] = df["n_artists"].apply(
        lambda x: "Solo (1 artist)" if x==1
        else ("Duo (2 artists)" if x==2
        else "Group (3+ artists)")
    )

    print(f"\n  {'Collaboration':<20} {'N':>8} {'Charted':>8} {'Chart%':>8}")
    print(f"  {'-'*48}")
    tier_data = {}
    for tier in ["Solo (1 artist)","Duo (2 artists)","Group (3+ artists)"]:
        tdf = df[df["collab_tier"]==tier]
        n=len(tdf); c=int(tdf["charted"].sum()); rate=c/n if n else 0
        bar = "█" * int(rate*300)
        print(f"  {tier:<20} {n:>8,} {c:>8,} {rate:>7.2%}  {bar}")
        tier_data[tier] = {"n":n,"charted":c,"chart_rate":round(float(rate),4)}

    results = {"tier_stats": tier_data, "ab_tests": []}
    solo  = df[df["collab_tier"]=="Solo (1 artist)"]["charted"]
    duo   = df[df["collab_tier"]=="Duo (2 artists)"]["charted"]
    group = df[df["collab_tier"]=="Group (3+ artists)"]["charted"]
    for la, ga, lb, gb in [
        ("Solo","Solo (1 artist)",   "Duo","Duo (2 artists)"),
        ("Solo","Solo (1 artist)",   "Group","Group (3+ artists)"),
        ("Duo", "Duo (2 artists)",   "Group","Group (3+ artists)"),
    ]:
        ga_data = df[df["collab_tier"]==ga]["charted"]
        gb_data = df[df["collab_tier"]==gb]["charted"]
        if len(ga_data)>30 and len(gb_data)>30:
            r = ab_test(ga_data, gb_data, ga, gb)
            print_ab(r)
            results["ab_tests"].append(r)
    return results

# ── 6. Song duration analysis ─────────────────────────────────────────────────

def duration_analysis(df):
    section("6. SONG DURATION vs CHART SUCCESS")

    df = df.copy()
    bins   = [0, 2, 2.5, 3, 3.5, 4, 5, 6, 100]
    labels = ["<2m","2–2.5m","2.5–3m","3–3.5m","3.5–4m","4–5m","5–6m","6m+"]
    df["duration_bucket"] = pd.cut(df["duration_min"], bins=bins, labels=labels)

    print(f"\n  {'Duration':<12} {'N':>8} {'Chart%':>8}  Bar")
    print(f"  {'-'*50}")
    bucket_stats = {}
    for label in labels:
        bdf  = df[df["duration_bucket"]==label]
        if len(bdf) < 20: continue
        rate = bdf["charted"].mean()
        bar  = "█" * int(rate*400)
        print(f"  {label:<12} {len(bdf):>8,} {rate:>7.2%}  {bar}")
        bucket_stats[label] = {"n":len(bdf),"chart_rate":round(float(rate),4)}

    # Optimal duration range
    best = max(bucket_stats.items(), key=lambda x: x[1]["chart_rate"])
    print(f"\n  ★  Optimal duration: {best[0]}  (chart rate {best[1]['chart_rate']:.2%})")

    # Correlation
    r, p = stats.pointbiserialr(df["duration_min"].fillna(0), df["charted"])
    print(f"  Duration ↔ charted correlation: r={r:+.4f}  p={p:.2e}")

    # A/B: short (<3.5m) vs long (>4.5m)
    short = df[df["duration_min"] <  3.5]["charted"]
    long_ = df[df["duration_min"] >= 4.5]["charted"]
    results = {"buckets": bucket_stats, "optimal": best[0],
               "correlation": {"r": round(float(r),4), "p": round(float(p),8)}}
    if len(short)>30 and len(long_)>30:
        r2 = ab_test(short, long_, "Short (<3.5 min)", "Long (>4.5 min)")
        print_ab(r2)
        results["ab_test"] = r2
    return results

# ── 7. Release timing ─────────────────────────────────────────────────────────

def release_timing_analysis(df):
    section("7. RELEASE TIMING — Optimal Month & Weekday")

    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    DAYS   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    results = {"by_month": {}, "by_weekday": {}}

    if "release_month" in df.columns:
        print(f"\n  Chart rate by release month:")
        print(f"  {'Month':<6} {'N':>8} {'Chart%':>8}  Bar")
        print(f"  {'-'*50}")
        month_stats = {}
        for m in range(1, 13):
            mdf  = df[df["release_month"]==m]
            if len(mdf) < 20: continue
            rate = mdf["charted"].mean()
            bar  = "█" * int(rate*400)
            print(f"  {MONTHS[m-1]:<6} {len(mdf):>8,} {rate:>7.2%}  {bar}")
            month_stats[MONTHS[m-1]] = {"n": len(mdf), "chart_rate": round(float(rate),4)}
        best_month = max(month_stats.items(), key=lambda x: x[1]["chart_rate"])
        print(f"\n  ★  Best release month: {best_month[0]}  ({best_month[1]['chart_rate']:.2%} chart rate)")
        results["by_month"]    = month_stats
        results["best_month"]  = best_month[0]

    if "release_dow" in df.columns:
        print(f"\n  Chart rate by release weekday:")
        print(f"  {'Day':<6} {'N':>8} {'Chart%':>8}  Bar")
        print(f"  {'-'*50}")
        day_stats = {}
        for d in range(7):
            ddf  = df[df["release_dow"]==d]
            if len(ddf) < 20: continue
            rate = ddf["charted"].mean()
            bar  = "█" * int(rate*400)
            print(f"  {DAYS[d]:<6} {len(ddf):>8,} {rate:>7.2%}  {bar}")
            day_stats[DAYS[d]] = {"n": len(ddf), "chart_rate": round(float(rate),4)}
        best_day = max(day_stats.items(), key=lambda x: x[1]["chart_rate"])
        print(f"\n  ★  Best release day: {best_day[0]}  ({best_day[1]['chart_rate']:.2%} chart rate)")
        results["by_weekday"] = day_stats
        results["best_day"]   = best_day[0]

    return results

# ── 8. Genre analysis ─────────────────────────────────────────────────────────

def genre_analysis(df, artists):
    section("8. GENRE CHART RATE ANALYSIS")

    artists_g = artists.copy()
    artists_g["genres_list"] = artists_g["genres"].apply(
        lambda s: ast.literal_eval(str(s)) if str(s).startswith("[") else []
    )
    exploded = artists_g.explode("genres_list")
    exploded = exploded[exploded["genres_list"].notna() & (exploded["genres_list"]!="")]

    merged = df.merge(exploded[["artist_id","genres_list"]],
                      left_on="primary_artist_id", right_on="artist_id", how="inner")
    pop_col = "popularity_track" if "popularity_track" in merged.columns else "popularity"
    agg = merged.groupby("genres_list").agg(
        n=("charted","count"), charted=("charted","sum"),
        avg_pop=(pop_col,"mean"),
    ).reset_index()
    agg["chart_rate"] = agg["charted"]/agg["n"]
    agg = agg[agg["n"]>=50].sort_values("chart_rate", ascending=False)

    print(f"\n  Top 15 genres by chart rate (min 50 tracks):")
    print(f"  {'Genre':<30} {'N':>7} {'Chart%':>8} {'Avg Pop':>9}")
    print(f"  {'-'*58}")
    for _, row in agg.head(15).iterrows():
        bar = "█" * int(row["chart_rate"]*30)
        print(f"  {row['genres_list']:<30} {row['n']:>7,} {row['chart_rate']:>7.2%} "
              f"{row['avg_pop']:>9.1f}  {bar}")

    print(f"\n  Bottom 5 genres:")
    for _, row in agg.tail(5).iterrows():
        print(f"  {row['genres_list']:<30} {row['n']:>7,} {row['chart_rate']:>7.2%}")

    return agg.to_dict(orient="records")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(save=True):
    log.info("Loading data…")
    tracks  = load_tracks()
    artists = load_artists()

    log.info("Joining artist metadata…")
    df = join_artists(tracks, artists)
    log.info("Joined dataset: %d rows", len(df))

    all_results = {}
    all_results["audio_vs_chart"]       = audio_vs_chart(df)
    all_results["hit_song_profile"]     = hit_song_profile(df)
    all_results["marketing_vs_music"]   = marketing_vs_music(df)
    all_results["emerging_artists"]     = emerging_artist_analysis(df)
    all_results["collaborations"]       = collaboration_analysis(df)
    all_results["duration"]             = duration_analysis(df)
    all_results["release_timing"]       = release_timing_analysis(df)
    all_results["genre_analysis"]       = genre_analysis(df, artists)

    if save:
        out = OUTPUT_DIR / "analysis_results.json"
        out.write_text(json.dumps(all_results, indent=2, default=str))
        log.info("Saved → %s", out)

    section("ANALYSIS COMPLETE ✓")
    print(f"  All results saved → {OUTPUT_DIR/'analysis_results.json'}")
    return all_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    run_analysis(save=True)
