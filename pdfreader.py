import argparse
import os
import sys

from PyPDF2 import PdfFileReader

DEBUG=False

def get_pdf_info(file_path: str):
    reader = PdfFileReader(file_path, strict=DEBUG, warndest=sys.stderr if DEBUG else os.devnull)
    standard_info = reader.documentInfo
    return {
        'page_count': reader.getNumPages(),
        # The problem here is that I don't know what's standard and what isn't.
        'creation_date': standard_info.get('/CreationDate'),
        'page_layout': reader.getPageLayout(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The PDF to read.')
    args = parser.parse_args()

    input_file = args.input_file
    info = get_pdf_info(input_file)
    print(info)


if __name__ == '__main__':
    main()
