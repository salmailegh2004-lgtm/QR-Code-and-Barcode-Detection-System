QR Code and Barcode Detection System
====================================

📁 Dataset Description

This folder contains a small set of test images used to evaluate the QR code and barcode detection system.

The dataset includes both QR codes and 1D barcodes under different visual conditions.

------------------------------------

📌 Files Description

1. BARCODE2.png
   - Standard 1D barcode with clear structure and visible digits.
   - Used to test basic barcode detection and decoding.

2. BARCODE3.png
   - Barcode with slight variations in thickness and spacing.
   - Used to evaluate robustness of gradient-based detection.

3. QR1.jpeg
   - Clean and standard QR code.
   - Easily detected using global QR detection.

4. QR2.jpeg
   - QR code with additional graphical elements (e.g., "SCAN ME" text and design).
   - More complex than QR1 and used to test robustness of detection.
   - May require local detection if global detection is affected by noise.

------------------------------------

📊 Purpose of the Dataset

The dataset is designed to evaluate:

- QR code detection using both global and local methods
- Barcode detection using gradient and morphological operations
- Decoding performance under different conditions
- System robustness to noise, design elements, and variations in image quality

------------------------------------

⚠️ Notes

- QR1.png is typically detected using global detection.
- QR2.png demonstrates cases where additional elements may affect detection.
- Barcode images are used to test the effectiveness of preprocessing and ROI extraction.

------------------------------------

👨‍💻 Usage

These images are used as input for the pipeline implemented in the Jupyter notebook and GUI application.

------------------------------------
