import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import detrend           
import pandas as pd                        
import csv                                 
import streamlit as st
import neurokit2 as nk
from collections import Counter
import math
VIDEO_PATH      = 'finger.mp4'
LOWCUT, HIGHCUT = 0.5, 4.0        
FILTER_ORDER    = 4               
MIN_PEAK_DIST   = 0.4             
PNN50_MS        = 50              

def run_ppg_pipeline(video_path: str):
    """
    Runs the full PPG analysis on a fingertip video file.
    Returns:
      metrics: dict with keys 'bpm','sdnn','rmssd','pnn50'
      figs:    dict with Matplotlib Figure objects for plots
    """
    cap = cv2.VideoCapture(video_path)      
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
    n_calib = int(5 * fps)
    baseline = raw_red[:n_calib].mean()
    TARGET_DC = 200
    gain = TARGET_DC / (baseline if baseline>0 else 1)
    raw_red = raw_red * gain                            
    raw_red = detrend(raw_red, type='constant')  
    nyq = 0.5 * fps
    b, a = butter(FILTER_ORDER, [LOWCUT/nyq, HIGHCUT/nyq], btype='band')
    filt = filtfilt(b, a, raw_red)
    times = np.arange(len(filt)) / fps

    rolling_std = (
        pd.Series(filt)
        .rolling(window=int(10*fps), center=True)
        .std()
        .bfill()              
        .ffill()              
        .values
)

    min_dist = int(MIN_PEAK_DIST * fps)
    peaks, _ = find_peaks(
        filt,
        distance=min_dist,
        prominence=rolling_std,      
        width=(0.2*fps, None)
    )

    window = int(0.15 * fps)            
    refined = []
    for p in peaks:
        start = max(p - window, 0)
        end   = min(p + window + 1, len(filt))
        loc   = start + np.argmax(filt[start:end])
        if filt[loc] >= (np.mean(filt) + 0.3*np.std(filt)):
            refined.append(loc)
    peaks = np.array(sorted(set(refined)))
    peak_times = peaks / fps

    start_idx = int(10 * fps)            
    filt_trimmed = filt[start_idx:]      
    times_trimmed = np.arange(len(filt_trimmed)) / fps
    peaks_rel = peaks[peaks >= start_idx] - start_idx
    peak_times = peaks_rel / fps

    ibis_all   = np.diff(peak_times)
    valid_ibis = ibis_all[(ibis_all > 0.4) & (ibis_all < 1.2)]
    ibis_ms     = valid_ibis * 1000
    ibis_clean  = ibis_ms[(ibis_ms >= 300) & (ibis_ms <= 2000)]
    median_ibi = np.median(ibis_clean)
    ibis_clean = ibis_clean[(ibis_clean >= 0.8*median_ibi) & (ibis_clean <= 1.2*median_ibi)]
    diffs_ms = np.abs(np.diff(ibis_clean))
    nn50_count = np.sum(diffs_ms > 50)
    rawpnn50 = (nn50_count / len(diffs_ms)) * 100
    pnn50 = math.sqrt(rawpnn50)*2
    duration = len(filt_trimmed) / fps
    bpm      = len(valid_ibis) / duration * 60
    sdnn     = np.std(valid_ibis) * 1000
    rmssd    = np.sqrt(np.mean(np.diff(valid_ibis)**2)) * 1000

    metrics = {
        'bpm':   float(bpm),  
        'sdnn':  float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50)
    }
    rhythm_label = classify_rhythm(valid_ibis,metrics)      
    metrics['rhythm'] = rhythm_label                 
    from scipy.signal import welch

    rri_ms    = valid_ibis * 1000                        # IBI in ms
    rri_times = (peak_times[1:] + peak_times[:-1]) / 2   # mid‐beat times 
    fs_interp = 4.0                                      # 4 Hz interpolation
    t_interp  = np.arange(rri_times[0], rri_times[-1], 1/fs_interp)
    if len(rri_times) != len(rri_ms):
        minlen = min(len(rri_times), len(rri_ms))
        rri_times = rri_times[:minlen]
        rri_ms    = rri_ms[:minlen]
    rri_interp = np.interp(t_interp, rri_times, rri_ms)


    f, Pxx = welch(rri_interp, fs=fs_interp, nperseg=min(256, len(rri_interp)))


    lf_band = (f >= 0.04) & (f <  0.15)
    hf_band = (f >= 0.15) & (f <= 0.40)
    lf_power = np.trapezoid(Pxx[lf_band], f[lf_band])
    hf_power = np.trapezoid(Pxx[hf_band], f[hf_band])


    metrics['lf_hf'] = float(lf_power / hf_power) if hf_power > 0 else np.nan

    if len(valid_ibis) < 20:
        sampen_value = 0.6 + (rmssd / 1000) * 0.6  
    else:
        sampen_raw = nk.entropy_sample(
            valid_ibis,
            dimension=2,
            r=0.2 * np.std(valid_ibis)
        )
        sampen = sampen_raw[0] if isinstance(sampen_raw, tuple) else sampen_raw
        if not np.isfinite(sampen):
            sampen = rmssd / 1000  
        sampen_value = sampen

    metrics['sampen'] = float(sampen_value)  

    figs = {}
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(times_trimmed, filt_trimmed, label='Filtered PPG (Red)')
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

    return metrics, figs

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
    diffs = np.diff(ibis) * 1000        
    nn50  = np.sum(np.abs(diffs) > 50)  
    pnn50 = 100 * nn50 / len(diffs) if len(diffs)>0 else 0
    
    sampen = None
    try:
        import neurokit2 as nk
        sampen = nk.entropy_sample(ibis, dimension=2, r=0.2*np.std(ibis))
        if isinstance(sampen, tuple):
            sampen = sampen[0]
    except Exception:
        sampen = None
    
    def detect_pattern(n):
        ratios = ibis / mean_rr
        flags  = ratios < 0.80
        idxs   = np.where(flags)[0]
        diffs = np.diff(idxs)
        if diffs.size == 0:
            return False
        ctr = Counter(diffs)
        most, count = ctr.most_common(1)[0]
        return abs(most - n) <= 1 and count >= len(idxs) * 0.65
    
    mean_ibi_ms = np.mean(ibis) * 1000
    hr     = metrics['bpm']       # heart rate in BPM
    sdnn   = metrics['sdnn']      # SDNN in ms
    rmssd  = metrics['rmssd']     # RMSSD in ms
    pnn50  = metrics['pnn50']     # pNN50 in %
    cv = sdnn / mean_ibi_ms    
    if hr < 25 or pnn50 < 1 or sdnn > 180 or rmssd > 250:
        return "Ventricular Fibrillation *** PLEASE SEEK MEDICAL HELP ***"
    if pnn50 < 1.8 or cv > 0.353 or sdnn < 20 or rmssd < 20:
        return "Atrial Fibrillation"
    if detect_pattern(2):
        return "Ventricular Bigeminy"
    if detect_pattern(3):
        return "Ventricular Trigeminy"
    if np.any(diffs < -600) and np.any(diffs > 600):
        return "Premature Ventricular Contractions"
    if hr < 50:
        return "Sinus Bradycardia"
    if hr > 100:
        return "Sinus Tachycardia"
    return "Normal Sinus Rhythm"

if __name__ == '__main__':
    import sys
    metrics, figs = run_ppg_pipeline(VIDEO_PATH)
    print("Demo metrics:", metrics)
    figs['filtered_peaks'].savefig('demo_filtered.png', dpi=150)
    figs['ibi_hist'].savefig('demo_ibi_hist.png', dpi=150)
    import matplotlib.pyplot as plt
    plt.close('all')

    sys.exit(0)

