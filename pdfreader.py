import argparse
import os
import sys

from PyPDF2 import PdfFileReader
from PyPDF2.generic import EncodedStreamObject
from PyPDF2.pdf import PageObject

DEBUG=False

def get_pdf_info(reader: PdfFileReader) -> dict:
    standard_info = reader.documentInfo
    return {
        'page_count': reader.getNumPages(),
        # The problem here is that I don't know what's standard and what isn't.
        'creation_date': standard_info.get('/CreationDate'),
        'page_layout': reader.getPageLayout(),
    }

def gather_pages(reader: PdfFileReader, page_count: int) -> list:
    my_page: PageObject = reader.getPage(10)
    from pprint import pprint
    page_x_object = my_page['/Resources']['/XObject'].getObject()
    for entry in page_x_object:
        if page_x_object[entry]['/Subtype'] == '/Image':
            encoded_image: EncodedStreamObject = page_x_object[entry]
            print(type(encoded_image))
            print(encoded_image.items())
            width: int = encoded_image['/Width']
            height: int = encoded_image['/Height']
            # Workaround for explosive NotImplementedError - grabs
            # raw image data via private property
            image_data = encoded_image._data
            image_color_space = encoded_image['/ColorSpace']
            image_mode = 'RGB' if image_color_space == '/DeviceRGB' else 'P'
            print(f"Found an image whose mode is {image_mode}, of size {width} x {height}")
    
    page_contents = my_page.getContents()
    # print(dir(page_contents))
    # print(page_contents)
    # print(page_contents.getXmpMetadata())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The PDF to read.')
    args = parser.parse_args()

    input_file = args.input_file

    reader = PdfFileReader(input_file, strict=DEBUG, warndest=sys.stderr if DEBUG else os.devnull)
    info = get_pdf_info(reader)
    print(info)
    page_count: int = info['page_count']
    gather_pages(reader, page_count)


if __name__ == '__main__':
    main()
