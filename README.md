# Oman Plant Disease Detection - Simple DIP Prototype

This is a small implementation for COSC 4354 Introduction to Digital Image Processing.
It detects suspected disease regions in plant leaf images using basic image processing.

## What it uses
- Python
- OpenCV
- NumPy
- Pillow
- Streamlit

## Main idea
1. Upload a leaf image.
2. Convert the image to HSV color space.
3. Segment the leaf area.
4. Detect yellow/brown/dark disease-like spots.
5. Clean the mask using morphological processing.
6. Calculate the percentage of suspected disease area.

## How to run on Windows
Open the folder in VS Code, then open Terminal and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open the local link shown in the terminal.

## Important note
This is a first screening prototype only. It does not identify the exact disease name and it is not a replacement for an agriculture specialist.
