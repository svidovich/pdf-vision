# pdf-vision

I'm starting this project because I have a book I want textified that's in PDF format, but it appears to just be scanned pages.

# Structure

You can have a `scratch/` folder to stash your items, and the gitignore will handle that.

# Dependencies

We require:

- PyPDF2 for reading PDFs.
- Pillow for handling images.
- Tesseract for OCR.
- OpenCV for image pre-processing.

# Getting Started

We recommend a virtual environment.

```
$ python3 -m venv venv
$ source venv/bin/activate
$ python3 -m pip install -r requirements.txt
```
