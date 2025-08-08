import cv2
import numpy as np
import matplotlib.pyplot as plt

# ——— 1) Open the video file ——————————————————————————————
VIDEO_PATH = 'finger.mp4'  # ← ensure this file is in your project folder
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video file: {VIDEO_PATH}")

# ——— 2) Read video properties ————————————————————————————
fps = cap.get(cv2.CAP_PROP_FPS)                 # frames per second
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps                    # total duration in seconds
print(f"Frames: {frame_count}, FPS: {fps:.1f}, Duration: {duration:.1f} s")

# ——— 3) Extract mean green‐channel per frame ————————————————————
green_series = []  # will hold one mean‐intensity value per frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # OpenCV frames are BGR; index 1 → green channel
    green = frame[:, :, 1]
    red   = frame[:, :, 2]
    green_series.append(red.mean())

cap.release()

# ——— 4) Convert to NumPy array & build time axis ————————————
green_series = np.array(green_series)
times = np.arange(len(green_series)) / fps  # time in seconds for each sample

dc = np.mean(green_series)                # baseline
ac = np.ptp(green_series)                 # peak‑to‑peak
snr = ac / dc * 100                       # percent
print(f"Baseline (DC): {dc:.1f}")
print(f"AC amplitude: {ac:.1f}")
print(f"SNR: {snr:.1f}%")

# ——— 5) Plot raw PPG waveform ——————————————————————————
plt.figure(figsize=(8, 4))
plt.plot(times, green_series, linewidth=1)
plt.title("Raw PPG Signal (Mean Green Intensity)")
plt.xlabel("Time (s)")
plt.ylabel("Mean Green Value")
plt.grid(True)
plt.tight_layout()

# Optional: save the plot to a file
plt.savefig("ppg_raw_signal.png", dpi=150)

# Show interactive plot window
plt.show()
