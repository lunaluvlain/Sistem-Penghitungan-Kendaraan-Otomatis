import cv2 as cv
import numpy as np
from scipy.spatial.distance import euclidean
import time

count_lurus = 0
track_kendaraan = []
COUNTING_LINE_COLOR = (0, 0, 255)
DEBUG_MODE = True

# PREPROCESSING 
def preprocess(frame, mog):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)

    fg_mask = mog.apply(gray)

    # ROI Masking 
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    pts_lurus = np.array([[200, 200], [600, 200], [600, height], [200, height]], np.int32)
    cv.fillPoly(mask, [pts_lurus], 255)
    fg_mask = cv.bitwise_and(fg_mask, fg_mask, mask=mask)

    # Noise Reduction 
    fg_mask = cv.medianBlur(fg_mask, 5)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=1)

    return fg_mask

# DETECTION & TRACKING
def detect_and_count(frame, fg_mask, frame_size):
    global count_lurus, track_kendaraan

    height, width = frame_size
    min_y_lurus = int(height * 0.75)

    # Contour Detection
    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    current_centroids = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 1500:
            continue

        x, y, w, h = cv.boundingRect(cnt)
        rasio_aspek = w / float(h)
        if not (0.6 < rasio_aspek < 4.0):
            continue

        centroid = (x + w // 2, y + h // 2)
        current_centroids.append(centroid)

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(frame, centroid, 4, (255, 0, 0), -1)

    # Tracking & Counting
    new_track = []
    for new_c in current_centroids:
        matched = False
        for old_c in track_kendaraan:
            if euclidean(new_c[:2], old_c[:2]) < 50:
                matched = True

                # Crossing Line Lurus (bawah horizontal)
                if old_c[1] < min_y_lurus <= new_c[1] and 200 < new_c[0] < 600:
                    count_lurus += 1

                new_track.append((new_c[0], new_c[1]))
                break

        if not matched:
            new_track.append((new_c[0], new_c[1]))

    track_kendaraan = new_track

    # Buat Garis
    cv.line(frame, (210, min_y_lurus), (600, min_y_lurus), (255, 0, 0), 2)   

    # Tampilan Count
    cv.putText(frame, f"Jumlah Kendaraan: {count_lurus}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if DEBUG_MODE:
        cv.putText(frame, f"Tracking: {len(track_kendaraan)}", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def main():
    cap = cv.VideoCapture('cars2.mp4')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Ambil FPS video
    fps = cap.get(cv.CAP_PROP_FPS)

    # full-screen mode
    cv.namedWindow('Vehicle Counting', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('Vehicle Counting', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    mog = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=500, detectShadows=False)

    # kontrol real-time
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - prev_time
        delay = max(int(1000 / fps) - int(elapsed_time * 1000), 1)

        frame = cv.resize(frame, (800, 600))
        fg_mask = preprocess(frame, mog)

        detect_and_count(frame, fg_mask, frame.shape[:2])

        cv.imshow('Vehicle Counting', frame)
        cv.imshow('Foreground Mask', fg_mask) #note rani: kalau mau lihat foreground mask, uncomment baris ini guys
        
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break

        prev_time = current_time

    cap.release()
    cv.destroyAllWindows()

    # EVALUASI AKURASI
    manual_count_lurus = 6

    system_count_lurus = count_lurus

    tp = min(system_count_lurus, manual_count_lurus)
    fp = max(system_count_lurus - manual_count_lurus, 0)
    fn = max(manual_count_lurus - system_count_lurus, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp / manual_count_lurus if manual_count_lurus > 0 else 0

    print(f"Perhitungan Manual: {manual_count_lurus}")
    print(f"Perhitungan Sistem: {system_count_lurus}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print(f"Akurasi (TP/Manual): {accuracy:.2f}")

if __name__ == "__main__":
    main()
