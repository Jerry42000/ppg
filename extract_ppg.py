import cv2

# Replace 'finger.mp4' with your sample video path
cap = cv2.VideoCapture('finger.mp4')
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video opened: {fps:.1f} FPS")

# Read first frame
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to read first frame")

# Extract green channel mean intensity
green = frame[:, :, 1]
print("Frame shape:", frame.shape)
print("Mean green intensity:", green.mean())