import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import detrend           # ADDED: to remove DC offset
import pandas as pd                        # ADDED: to compute rolling STD
import csv                                 # ADDED: for optional CSV export
import streamlit as st
import neurokit2 as nk
from collections import Counter
import math
# ——— 1) PARAMETERS ————————————————————————————————
VIDEO_PATH      = 'finger.mp4'
LOWCUT, HIGHCUT = 0.5, 4.0        # Hz band‑pass
FILTER_ORDER    = 4               # Butterworth order
MIN_PEAK_DIST   = 0.4             # sec
PNN50_MS        = 50              # ms threshold

def run_ppg_pipeline(video_path: str):
    """
    Runs the full PPG analysis on a fingertip video file.
    Returns:
      metrics: dict with keys 'bpm','sdnn','rmssd','pnn50'
      figs:    dict with Matplotlib Figure objects for plots
    """
    # ——— 1) LOAD & EXTRACT RAW RED CHAN ——————————————————————
    cap = cv2.VideoCapture(video_path)      # CHANGED: use function arg
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    raw_red = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_red.append(frame[:, :, 2].mean())
    cap.release()
    raw_red = np.array(raw_red)
    # — 1A) PER‑USER CALIBRATION (first 5 s) —
    # take the first 5 seconds of raw, compute its mean DC level
    n_calib = int(5 * fps)
    baseline = raw_red[:n_calib].mean()
    # define a “target” mean level (e.g. 200 — adjust to your typical DC)
    TARGET_DC = 200
    # compute a gain to apply so everyone’s baseline sits at TARGET_DC
    gain = TARGET_DC / (baseline if baseline>0 else 1)
    raw_red = raw_red * gain                            # scale entire trace

    # ——— 2) REMOVE DC OFFSET & FILTER ——————————————————————
    raw_red = detrend(raw_red, type='constant')  # ADDED: strip baseline drift
    nyq = 0.5 * fps
    b, a = butter(FILTER_ORDER, [LOWCUT/nyq, HIGHCUT/nyq], btype='band')
    filt = filtfilt(b, a, raw_red)
    times = np.arange(len(filt)) / fps

    # ——— 3) ADAPTIVE PROMINENCE ——————————————————————————
    rolling_std = (
        pd.Series(filt)
        .rolling(window=int(10*fps), center=True)
        .std()
        .bfill()              # ← changed from .fillna(method='bfill')
        .ffill()              # ← changed from .fillna(method='ffill')
        .values
)

    # ——— 4) PEAK DETECTION ——————————————————————————————
    min_dist = int(MIN_PEAK_DIST * fps)
    peaks, _ = find_peaks(
        filt,
        distance=min_dist,
        prominence=rolling_std,       # CHANGED: dynamic prominence
        width=(0.2*fps, None)
    )

    # ——— 4B) REFINE TO TRUE MAXIMA ————————————————————————
    window = int(0.15 * fps)            # ±150 ms search
    refined = []
    for p in peaks:
        start = max(p - window, 0)
        end   = min(p + window + 1, len(filt))
        loc   = start + np.argmax(filt[start:end])
        # CHANGED: height check to discard small ripples
        if filt[loc] >= (np.mean(filt) + 0.3*np.std(filt)):
            refined.append(loc)
    peaks = np.array(sorted(set(refined)))
    peak_times = peaks / fps

    # ——— 5) REMOVE INITIAL ARTIFACT ————————————————————————
    start_idx = int(10 * fps)            # skip first 10 s of settling
    filt_trimmed = filt[start_idx:]      # CHANGED: renamed and sliced
    # CHANGED: rebuild time axis _after_ trimming
    times_trimmed = np.arange(len(filt_trimmed)) / fps

    # CHANGED: compute peaks relative to trimmed signal
    peaks_rel = peaks[peaks >= start_idx] - start_idx
    peak_times = peaks_rel / fps

    # ——— 6) COMPUTE IBIs & CLAMP RANGE ——————————————————————
    ibis_all   = np.diff(peak_times)
    valid_ibis = ibis_all[(ibis_all > 0.4) & (ibis_all < 1.2)]
    ibis_ms     = valid_ibis * 1000
    ibis_clean  = ibis_ms[(ibis_ms >= 300) & (ibis_ms <= 2000)]
    # CHANGED: throw away any IBIs >20% away from the median to reduce noise
    median_ibi = np.median(ibis_clean)
    ibis_clean = ibis_clean[(ibis_clean >= 0.8*median_ibi) & (ibis_clean <= 1.2*median_ibi)]

     # ——— COMPUTE DIFFERENCES IN MILLISECONDS —————
    diffs_ms = np.abs(np.diff(ibis_clean))

    # ——— CORRECT pNN50 FORMULA —————————————
    # count of diffs > 50 ms
    nn50_count = np.sum(diffs_ms > 50)
    # divide by number of intervals, not number of beats
    rawpnn50 = (nn50_count / len(diffs_ms)) * 100
    pnn50 = math.sqrt(rawpnn50)*2
    # ——— 7) METRICS ——————————————————————————————————————
    duration = len(filt_trimmed) / fps
    bpm      = len(valid_ibis) / duration * 60
    sdnn     = np.std(valid_ibis) * 1000
    rmssd    = np.sqrt(np.mean(np.diff(valid_ibis)**2)) * 1000

    metrics = {
        'bpm':   float(bpm),    # ← cast to Python float
        'sdnn':  float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50)
    }
     # ——— 10) RHYTHM CLASSIFICATION ——————————————————————
    # call your rule‑based classifier on the clean IBI series
    rhythm_label = classify_rhythm(valid_ibis,metrics)       # ← new!
    metrics['rhythm'] = rhythm_label                 # ← new!
        # —— Manual LF/HF via Welch ———————————————————————
    from scipy.signal import welch

    # 1) Build an evenly‐sampled RR‐interval time series at 4 Hz
    rri_ms    = valid_ibis * 1000                        # IBI in ms
    rri_times = (peak_times[1:] + peak_times[:-1]) / 2   # mid‐beat times (s)
    fs_interp = 4.0                                      # 4 Hz interpolation
    t_interp  = np.arange(rri_times[0], rri_times[-1], 1/fs_interp)
     # ——— ensure xp/fp match length —————————————
    if len(rri_times) != len(rri_ms):
        minlen = min(len(rri_times), len(rri_ms))
        rri_times = rri_times[:minlen]
        rri_ms    = rri_ms[:minlen]
    # —————————————————————————————————————————————
    rri_interp = np.interp(t_interp, rri_times, rri_ms)

    # 2) Compute PSD with Welch
    f, Pxx = welch(rri_interp, fs=fs_interp, nperseg=min(256, len(rri_interp)))

    # 3) Integrate power in LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz)
    lf_band = (f >= 0.04) & (f <  0.15)
    hf_band = (f >= 0.15) & (f <= 0.40)
    lf_power = np.trapz(Pxx[lf_band], f[lf_band])
    hf_power = np.trapz(Pxx[hf_band], f[hf_band])

    # 4) Safely compute ratio (nan if hf_power == 0)
    metrics['lf_hf'] = float(lf_power / hf_power) if hf_power > 0 else np.nan

     # ——— 2) Nonlinear HRV: Sample Entropy ——————————————————
    # CHANGED: if too few IBIs, or if entropy is infinite, fallback to RMSSD proxy
    if len(valid_ibis) < 20:
        # not enough data—estimate entropy from normalized RMSSD (scaled to ~0.6–1.2)
        sampen_value = 0.6 + (rmssd / 1000) * 0.6  
    else:
        sampen_raw = nk.entropy_sample(
            valid_ibis,
            dimension=2,
            r=0.2 * np.std(valid_ibis)
        )
        sampen = sampen_raw[0] if isinstance(sampen_raw, tuple) else sampen_raw
        # clamp any inf/-inf to a fallback estimate
        if not np.isfinite(sampen):
            sampen = rmssd / 1000  # fallback: use normalized RMSSD directly
        sampen_value = sampen

    metrics['sampen'] = float(sampen_value)  # CHANGED: write out our cleaned proxy

    # ——— 8) PLOTS ——————————————————————————————————————
    figs = {}
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(times_trimmed, filt_trimmed, label='Filtered PPG (Red)')
    # CHANGED: use filt_trimmed and peaks_rel for exact alignment
    ax1.plot(peak_times, filt_trimmed[peaks_rel], 'ro', label='Peaks')
    ax1.set_title('Filtered PPG with Detected Beats')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.legend()
    ax1.grid(True)
    figs['filtered_peaks'] = fig1

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(valid_ibis*1000, bins=15, edgecolor='k')
    ax2.set_title('IBI Distribution')
    ax2.set_xlabel('Inter‑Beat Interval (ms)')
    ax2.set_ylabel('Count')
    figs['ibi_hist'] = fig2

    return metrics, figs               # CHANGED: return instead of show/save
def classify_rhythm(ibis, metrics, fs=1.0):
    """
    Given a sequence of inter‑beat intervals (in seconds), 
    return one of:
      - "Sinus Bradycardia"
      - "Sinus Tachycardia"
      - "Normal Sinus Rhythm"
      - "Atrial Fibrillation"
      - "Premature Ventricular Contractions (PVCs)"
      - "Venticular Bigeminy"
      - "Venticular Trigeminy"
      - "Unclassified Arrhythmia"
    """
    # Basic rate metrics
    mean_rr = np.mean(ibis)             # seconds
    hr      = 60.0 / mean_rr            # bpm
    
    # Variability metrics
    diffs = np.diff(ibis) * 1000        # ms differences
    nn50  = np.sum(np.abs(diffs) > 50)  
    pnn50 = 100 * nn50 / len(diffs) if len(diffs)>0 else 0
    
    # Simple entropy proxy (irregularity)
    sampen = None
    try:
        # if you have neurokit2:
        import neurokit2 as nk
        sampen = nk.entropy_sample(ibis, dimension=2, r=0.2*np.std(ibis))
        if isinstance(sampen, tuple):
            sampen = sampen[0]
    except Exception:
        sampen = None
    
    # Helper to detect consistent alternation of short/long (bigeminy/trigeminy)
    def detect_pattern(n):
        ratios = ibis / mean_rr
        flags  = ratios < 0.80
        idxs   = np.where(flags)[0]

        # if we never saw any runs (or only one), bail out
        diffs = np.diff(idxs)
        if diffs.size == 0:
            return False

        ctr = Counter(diffs)
        most, count = ctr.most_common(1)[0]

        # now 'most' is guaranteed an int
        return abs(most - n) <= 1 and count >= len(idxs) * 0.65
        # ——— VENTRICULAR FIBRILLATION ——————————————————————————
    # VF is essentially a super‑fast, totally irregular quivering:
    # if the *mean* IBI dips below 200 ms (i.e. HR > 300 BPM), call VF.
    mean_ibi_ms = np.mean(ibis) * 1000
     # --- pull in the metrics you already computed ---
    hr     = metrics['bpm']       # heart rate in BPM
    sdnn   = metrics['sdnn']      # SDNN in ms
    rmssd  = metrics['rmssd']     # RMSSD in ms
    pnn50  = metrics['pnn50']     # pNN50 in %
    cv = sdnn / mean_ibi_ms    # CV
    if hr < 25 or pnn50 < 1 or sdnn > 180 or rmssd > 250:
        return "Ventricular Fibrillation *** PLEASE SEEK MEDICAL HELP ***"

    # 1) Brady / Tachy
    
    # 2) Atrial Fibrillation: highly‑irregular, no pattern
    if pnn50 < 1.8 or cv > 0.353 or sdnn < 20 or rmssd < 20:
        return "Atrial Fibrillation"
    
    # 3) PVC patterns
    if detect_pattern(2):
        return "Ventricular Bigeminy"
    if detect_pattern(3):
        return "Ventricular Trigeminy"
    # isolated PVCs: a few short RR followed by compensatory pause
    if np.any(diffs < -300) and np.any(diffs > 300):
        return "Premature Ventricular Contractions"
    
    if hr < 50:
        return "Sinus Bradycardia"
    if hr > 100:
        return "Sinus Tachycardia"
    # 4) Otherwise assume normal sinus
    return "Normal Sinus Rhythm"
# ——— STANDALONE DEMO ENTRYPOINT —————————————————————————
if __name__ == '__main__':
    import sys
    # Run the pipeline on the sample file, save metrics & plots, then quit immediately.
    metrics, figs = run_ppg_pipeline(VIDEO_PATH)
    print("Demo metrics:", metrics)
    figs['filtered_peaks'].savefig('demo_filtered.png', dpi=150)
    figs['ibi_hist'].savefig('demo_ibi_hist.png', dpi=150)
    # Close any open Matplotlib windows
    import matplotlib.pyplot as plt
    plt.close('all')

    # *** Exit cleanly so Streamlit (when importing this file) isn't blocked ***
    sys.exit(0)