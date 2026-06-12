"""
spotify_analysis/src/collector.py
──────────────────────────────────
Spotify Audio Features Data Collector
Pulls 100K+ tracks across genres via Spotify Web API using spotipy.

Strategy:
  1. Seed from curated genre playlists (Spotify's own editorial playlists)
  2. Expand via Related Artists graph traversal
  3. Pull every track's Audio Features + Track metadata in batches
  4. Label: popularity >= CHART_THRESHOLD → charted = 1
"""

import os
import time
import logging
import random
from pathlib import Path
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CHART_THRESHOLD   = 70          # popularity score ≥ this → "charted"
TARGET_TRACKS     = 110_000     # stop collecting after this many unique tracks
BATCH_SIZE        = 50          # Spotify allows up to 100 IDs per audio-features call
RATE_LIMIT_SLEEP  = 0.1         # seconds between API calls (stay under rate limits)
MAX_RETRIES       = 3           # retries on transient errors
RELATED_ARTISTS_DEPTH = 2       # how many hops to traverse in the artist graph

DATA_DIR   = Path(__file__).parent.parent / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Genre seed playlists ──────────────────────────────────────────────────────
# These are Spotify's own editorial "This Is" / genre flagship playlists.
# IDs are stable; update if Spotify deprecates any.

GENRE_PLAYLISTS = {
    "pop":          "37i9dQZF1DXcBWIGoYBM5M",   # Today's Top Hits
    "hip_hop":      "37i9dQZF1DX0XUsuxWHRQd",   # RapCaviar
    "edm":          "37i9dQZF1DX4dyzvuaRJ0n",   # mint
    "r_and_b":      "37i9dQZF1DX4SBhb3fqCJd",   # Are & Be
    "rock":         "37i9dQZF1DXcF6B6QPhFDv",   # Rock This
    "latin":        "37i9dQZF1DX10zKzsJ2jva",   # Viva Latino
    "country":      "37i9dQZF1DX1lVhptIYRda",   # Hot Country
    "classical":    "37i9dQZF1DWWEJlAGA9gs0",   # Classical Essentials
    "jazz":         "37i9dQZF1DXbITWG1ZJKYt",   # Jazz Classics
    "metal":        "37i9dQZF1DWTcqUzwhNmKv",   # Metal Essentials
    "indie":        "37i9dQZF1DX2Nc3B70tvx0",   # Indie Pop
    "soul":         "37i9dQZF1DWTx0xog3gB3H",   # Soul Music
    "funk":         "37i9dQZF1DWWvvyNmW9V9a",   # Funk Outta Here
    "reggae":       "37i9dQZF1DWYBO1MoTDhZI",   # Reggae Classics
    "blues":        "37i9dQZF1DXd9rSDyQguIk",   # Blues Classics
    "k_pop":        "37i9dQZF1DX9tPFwDMOswN",   # K-Pop Daebak
    "afrobeats":    "37i9dQZF1DWYkaDif7Ztbp",   # Afro Party
    "punk":         "37i9dQZF1DXd6tJtr4qeot",   # Punk Classics
    "folk":         "37i9dQZF1DX4OzrY981I1W",   # Folk & Friends
    "ambient":      "37i9dQZF1DX3Iej0Xt9oqU",   # Ambient Chill
    "gospel":       "37i9dQZF1DXcb6CghgBBBN",   # Holy Hip Hop
    "house":        "37i9dQZF1DX6J5NfMJS675",   # Deep House Relax
    "trap":         "37i9dQZF1DWY4xHMTAC9KE",   # Trap Nation
    "emo":          "37i9dQZF1DX6FFEKiCMFCz",   # Emo Nite
    "dancehall":    "37i9dQZF1DX8SfyqmSFDwe",   # Dancehall Official
    "new_age":      "37i9dQZF1DX8ymr6UES7vc",   # New Age
}

# ── Spotify client ────────────────────────────────────────────────────────────

def get_spotify_client() -> spotipy.Spotify:
    """
    Build authenticated Spotify client.
    Reads SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from environment.
    Get credentials at: https://developer.spotify.com/dashboard
    """
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET.\n"
            "Export them before running:\n"
            "  export SPOTIFY_CLIENT_ID='your_id'\n"
            "  export SPOTIFY_CLIENT_SECRET='your_secret'"
        )

    auth = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    return spotipy.Spotify(auth_manager=auth, requests_timeout=10)


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, *args, retries=MAX_RETRIES, **kwargs):
    """Call fn(*args, **kwargs), retrying on rate-limit or transient errors."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:                         # rate limited
                wait = int(e.headers.get("Retry-After", 5))
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
            elif e.http_status >= 500:                       # server error
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                log.warning("Transient error (%s), retrying…", e)
            else:
                raise
    return None


# ── Playlist → track IDs ──────────────────────────────────────────────────────

def fetch_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> list[str]:
    """Return all track IDs from a playlist (handles pagination)."""
    track_ids = []
    offset = 0
    while True:
        resp = _with_retry(
            sp.playlist_items, playlist_id,
            fields="items(track(id)),next",
            limit=100, offset=offset,
        )
        if not resp:
            break
        items = resp.get("items", [])
        for item in items:
            t = item.get("track")
            if t and t.get("id"):
                track_ids.append(t["id"])
        if not resp.get("next"):
            break
        offset += 100
        time.sleep(RATE_LIMIT_SLEEP)
    return track_ids


# ── Artist → related artists ──────────────────────────────────────────────────

def expand_via_related_artists(
    sp: spotipy.Spotify,
    seed_artist_ids: list[str],
    depth: int = RELATED_ARTISTS_DEPTH,
    max_artists: int = 5_000,
) -> set[str]:
    """
    BFS over the related-artists graph starting from seed_artist_ids.
    Returns a set of all discovered artist IDs.
    """
    visited   = set(seed_artist_ids)
    frontier  = list(seed_artist_ids)

    for _ in range(depth):
        next_frontier = []
        random.shuffle(frontier)                             # avoid clustering
        for artist_id in frontier[:500]:                     # cap per depth
            if len(visited) >= max_artists:
                break
            related = _with_retry(sp.artist_related_artists, artist_id)
            if not related:
                continue
            for a in related.get("artists", []):
                aid = a.get("id")
                if aid and aid not in visited:
                    visited.add(aid)
                    next_frontier.append(aid)
            time.sleep(RATE_LIMIT_SLEEP)
        frontier = next_frontier
        log.info("  Related-artist expansion: %d artists discovered", len(visited))
        if not frontier:
            break

    return visited


# ── Artist → top tracks ───────────────────────────────────────────────────────

def fetch_artist_top_tracks(
    sp: spotipy.Spotify,
    artist_ids: set[str],
    market: str = "US",
) -> list[str]:
    """Return track IDs for the top-10 tracks of each artist."""
    track_ids = []
    for i, artist_id in enumerate(artist_ids):
        resp = _with_retry(sp.artist_top_tracks, artist_id, country=market)
        if resp:
            for t in resp.get("tracks", []):
                if t.get("id"):
                    track_ids.append(t["id"])
        if i % 200 == 0:
            log.info("  Top-tracks: processed %d/%d artists", i, len(artist_ids))
        time.sleep(RATE_LIMIT_SLEEP)
    return track_ids


# ── Track metadata ────────────────────────────────────────────────────────────

def fetch_track_metadata(
    sp: spotipy.Spotify,
    track_ids: list[str],
) -> dict[str, dict]:
    """
    Fetch track metadata (name, artists, album, popularity, release date)
    in batches of 50.  Returns dict keyed by track_id.
    """
    meta = {}
    for i in range(0, len(track_ids), BATCH_SIZE):
        batch = track_ids[i : i + BATCH_SIZE]
        resp  = _with_retry(sp.tracks, batch)
        if not resp:
            continue
        for t in resp.get("tracks", []) or []:
            if not t or not t.get("id"):
                continue
            artists = [a["name"] for a in t.get("artists", [])]
            meta[t["id"]] = {
                "track_id":       t["id"],
                "track_name":     t.get("name", ""),
                "artist_names":   ", ".join(artists),
                "artist_ids":     ", ".join(a["id"] for a in t.get("artists", [])),
                "album_name":     t.get("album", {}).get("name", ""),
                "release_date":   t.get("album", {}).get("release_date", ""),
                "popularity":     t.get("popularity", 0),
                "explicit":       int(t.get("explicit", False)),
                "duration_ms":    t.get("duration_ms", 0),
                "markets_count":  len(t.get("available_markets", [])),
            }
        if i % 5000 == 0 and i > 0:
            log.info("  Metadata: %d/%d tracks fetched", i, len(track_ids))
        time.sleep(RATE_LIMIT_SLEEP)
    return meta


# ── Audio features ────────────────────────────────────────────────────────────

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature",
]

def fetch_audio_features(
    sp: spotipy.Spotify,
    track_ids: list[str],
) -> dict[str, dict]:
    """
    Fetch audio features for up to 100 tracks per call.
    Returns dict keyed by track_id.
    """
    features = {}
    for i in range(0, len(track_ids), 100):               # API max = 100
        batch = track_ids[i : i + 100]
        resp  = _with_retry(sp.audio_features, batch)
        if not resp:
            continue
        for af in resp or []:
            if not af or not af.get("id"):
                continue
            features[af["id"]] = {k: af.get(k) for k in AUDIO_FEATURES}
        if i % 10_000 == 0 and i > 0:
            log.info("  Audio features: %d/%d tracks fetched", i, len(track_ids))
        time.sleep(RATE_LIMIT_SLEEP)
    return features


# ── Combine & label ───────────────────────────────────────────────────────────

def build_dataframe(
    meta: dict[str, dict],
    audio: dict[str, dict],
    genre_map: dict[str, str],
) -> pd.DataFrame:
    """
    Join metadata + audio features.
    Label: charted = 1 if popularity >= CHART_THRESHOLD.
    """
    rows = []
    for tid, m in meta.items():
        af = audio.get(tid)
        if not af:
            continue
        row = {**m, **af}
        row["genre"]   = genre_map.get(tid, "unknown")
        row["charted"] = int(m["popularity"] >= CHART_THRESHOLD)
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- derived features
    df["release_year"]  = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["duration_min"]  = (df["duration_ms"] / 60_000).round(2)
    df["energy_dance"]  = (df["energy"] * df["danceability"]).round(4)   # interaction term
    df["loud_energy"]   = (df["loudness"].abs() * df["energy"]).round(4) # interaction term

    # Spotify loudness is negative dBFS; normalise to [0, 1]
    loud_min, loud_max  = -60.0, 0.0
    df["loudness_norm"] = ((df["loudness"] - loud_min) / (loud_max - loud_min)).clip(0, 1).round(4)

    # Tempo: normalise to [0, 1] (typical range 40–220 BPM)
    df["tempo_norm"]    = ((df["tempo"] - 40) / 180).clip(0, 1).round(4)

    return df.drop_duplicates(subset="track_id").reset_index(drop=True)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(df: pd.DataFrame, name: str = "checkpoint"):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("Checkpoint saved → %s  (%d rows)", path, len(df))


def load_checkpoint(name: str = "checkpoint") -> Optional[pd.DataFrame]:
    path = RAW_DIR / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        log.info("Checkpoint loaded ← %s  (%d rows)", path, len(df))
        return df
    return None


# ── Main orchestrator ─────────────────────────────────────────────────────────

def collect(resume: bool = True) -> pd.DataFrame:
    """
    Full collection pipeline.

    Args:
        resume: if True, load existing checkpoint and only collect new tracks.

    Returns:
        DataFrame with all tracks + audio features + labels.
    """
    sp = get_spotify_client()
    log.info("Spotify client authenticated ✓")

    existing_ids: set[str] = set()
    all_meta:  dict = {}
    all_audio: dict = {}
    genre_map: dict[str, str] = {}

    if resume:
        ckpt = load_checkpoint()
        if ckpt is not None:
            existing_ids = set(ckpt["track_id"].tolist())
            log.info("Resuming — %d tracks already collected", len(existing_ids))

    # ── Phase 1: Playlist seeds ───────────────────────────────────────────────
    log.info("Phase 1 — collecting from %d genre playlists…", len(GENRE_PLAYLISTS))
    all_track_ids: set[str] = set()

    for genre, playlist_id in GENRE_PLAYLISTS.items():
        ids = fetch_playlist_tracks(sp, playlist_id)
        new_ids = [i for i in ids if i not in existing_ids]
        for tid in new_ids:
            genre_map[tid] = genre
        all_track_ids.update(new_ids)
        log.info("  %-12s  +%d tracks  (total %d)", genre, len(new_ids), len(all_track_ids))

    # ── Phase 2: Related-artist expansion ────────────────────────────────────
    log.info("Phase 2 — expanding via related artists…")
    seed_artist_ids: list[str] = []
    for tid_batch in [list(all_track_ids)[i:i+50] for i in range(0, min(500, len(all_track_ids)), 50)]:
        resp = _with_retry(sp.tracks, tid_batch)
        if resp:
            for t in resp.get("tracks", []) or []:
                if t:
                    for a in t.get("artists", []):
                        if a.get("id"):
                            seed_artist_ids.append(a["id"])
        time.sleep(RATE_LIMIT_SLEEP)

    seed_artist_ids = list(set(seed_artist_ids))
    log.info("  Seed artists: %d", len(seed_artist_ids))

    related_artists = expand_via_related_artists(sp, seed_artist_ids)
    new_track_ids   = fetch_artist_top_tracks(sp, related_artists)

    for tid in new_track_ids:
        if tid not in genre_map:
            genre_map[tid] = "mixed"
    all_track_ids.update(t for t in new_track_ids if t not in existing_ids)

    log.info("After expansion: %d unique track IDs", len(all_track_ids))

    # ── Trim to target ────────────────────────────────────────────────────────
    id_list = list(all_track_ids)
    random.shuffle(id_list)
    id_list = id_list[:TARGET_TRACKS]
    log.info("Collecting features for %d tracks…", len(id_list))

    # ── Phase 3: Metadata ─────────────────────────────────────────────────────
    log.info("Phase 3 — fetching track metadata…")
    all_meta = fetch_track_metadata(sp, id_list)
    log.info("  Metadata fetched for %d tracks", len(all_meta))

    # Checkpoint after metadata
    interim_df = pd.DataFrame(all_meta.values())
    save_checkpoint(interim_df, "meta_checkpoint")

    # ── Phase 4: Audio features ───────────────────────────────────────────────
    log.info("Phase 4 — fetching audio features…")
    all_audio = fetch_audio_features(sp, list(all_meta.keys()))
    log.info("  Audio features fetched for %d tracks", len(all_audio))

    # ── Phase 5: Build final dataset ─────────────────────────────────────────
    log.info("Phase 5 — building final dataset…")
    df = build_dataframe(all_meta, all_audio, genre_map)

    # Merge with previous checkpoint if resuming
    if resume and existing_ids:
        ckpt = load_checkpoint()
        if ckpt is not None:
            df = pd.concat([ckpt, df]).drop_duplicates(subset="track_id").reset_index(drop=True)

    log.info("Final dataset: %d rows, %d columns", len(df), len(df.columns))
    log.info("Charted tracks: %d (%.1f%%)", df["charted"].sum(), df["charted"].mean() * 100)

    save_checkpoint(df)

    # Save final CSV for inspection
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROC_DIR / "spotify_tracks.csv"
    df.to_csv(csv_path, index=False)
    log.info("Final CSV saved → %s", csv_path)

    return df


if __name__ == "__main__":
    df = collect(resume=True)
    print(df.head())
    print("\nColumn dtypes:")
    print(df.dtypes)
    print(f"\nClass balance:\n{df['charted'].value_counts()}")
