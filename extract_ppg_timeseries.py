import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH = 'finger.mp4'  
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)                 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps                    
print(f"Frames: {frame_count}, FPS: {fps:.1f}, Duration: {duration:.1f} s")

green_series = []  #

while True:
    ret, frame = cap.read()
    if not ret:
        break
    green = frame[:, :, 1]
    red   = frame[:, :, 2]
    green_series.append(red.mean())

cap.release()

green_series = np.array(green_series)
times = np.arange(len(green_series)) / fps  

dc = np.mean(green_series)                
ac = np.ptp(green_series)                 
snr = ac / dc * 100                       
print(f"Baseline (DC): {dc:.1f}")
print(f"AC amplitude: {ac:.1f}")
print(f"SNR: {snr:.1f}%")


plt.figure(figsize=(8, 4))
plt.plot(times, green_series, linewidth=1)
plt.title("Raw PPG Signal (Mean Green Intensity)")
plt.xlabel("Time (s)")
plt.ylabel("Mean Green Value")
plt.grid(True)
plt.tight_layout()


plt.savefig("ppg_raw_signal.png", dpi=150)

plt.show()
