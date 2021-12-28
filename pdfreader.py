import argparse
import os
import sys

from PyPDF2 import PdfFileReader
from PyPDF2.generic import EncodedStreamObject
from PyPDF2.pdf import PageObject

DEBUG = False

filter_to_extension = {
    '/FlateDecode': '.png',
    '/DCTDecode': '.jpg',
    '/JPXDecode': '.jp2',
}

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
    # For now, let's put the pages as images into this list and return it
    output_list = list()
    page_x_object = my_page['/Resources']['/XObject'].getObject()
    for entry in page_x_object:
        if page_x_object[entry]['/Subtype'] == '/Image':
            encoded_image: EncodedStreamObject = page_x_object[entry]

            # Workaround for explosive NotImplementedError - grabs
            # raw image data via private property
            image_data: bytes = encoded_image._data
            # TODO: Right now we're doing a lot of hard dictionary accesses
            # Later this should probably be a little safer
            width: int = encoded_image['/Width']
            height: int = encoded_image['/Height']

            image_filter: str = encoded_image['/Filter']
            image_file_extension: str = filter_to_extension[image_filter]

            image_color_space = encoded_image['/ColorSpace']
            image_mode = 'RGB' if image_color_space == '/DeviceRGB' else 'P'
            print(f"Found an image whose mode is {image_mode}, of size {width} x {height}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The PDF to read.')
    args = parser.parse_args()

    input_file = args.input_file

    reader = PdfFileReader(input_file, strict=DEBUG, warndest=sys.stderr if DEBUG else os.devnull)
    info = get_pdf_info(reader)
    page_count: int = info['page_count']
    gather_pages(reader, page_count)


if __name__ == '__main__':
    main()
