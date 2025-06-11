# Sistem-Penghitungan-Kendaraan-Otomatis
DeepCount adalah sistem berbasis video untuk mendeteksi dan menghitung kendaraan roda empat atau lebih secara otomatis, akurat, dan real-time, tanpa memerlukan sensor fisik tambahan. Proyek ini memanfaatkan teknik pengolahan citra digital konvensional seperti background subtraction dan centroid tracking.

### Dataset
Dataset berupa video rekaman jalan raya yang digunakan untuk deteksi dan pelacakan kendaraan. File video disimpan dalam folder *dataset* di dalam repository ini. Video ini berisi lalu lintas kendaraan dari perspektif statis (kamera pengawas jalan)

### Tujuan
Membangun sistem penghitungan kendaraan dari video lalu lintas secara efisien tanpa perangkat keras tambahan menggunakan metode konvensional.

### Metode
1. Preprocessing: Grayscale, Blur, dan ROI
2. Segmentasi: MOG2 + Operasi Morfologi
3. Deteksi & Tracking: Contour + Centroid + Euclidean Distance
4. Counting: Saat centroid melewati garis horizontal
5. Evaluasi: Precision, Recall, F1-Score, Akurasi

### Result
![Screenshot 2025-06-10 151841](https://github.com/user-attachments/assets/d55d4df9-34b8-4804-96f3-667bf44d5607)
Implementasi sistem berhasil mendeteksi dan menghitung kendaraan secara akurat pada dataset uji. Dari total 6 kendaraan yang tampil dalam video, seluruhnya berhasil dikenali oleh sistem (True Positives = 6, False Negatives = 0). Tidak terdapat kesalahan deteksi atau deteksi palsu. Evaluasi kinerja menunjukkan nilai Precision, Recall, F1-Score, dan Akurasi masing-masing sebesar 1.00, yang mengindikasikan performa sistem yang sangat optimal terhadap dataset yang digunakan
