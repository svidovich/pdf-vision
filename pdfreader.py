import argparse
import os
import sys

from io import BytesIO
from typing import List

from PIL import Image

from PyPDF2 import PdfFileReader
from PyPDF2.generic import DictionaryObject, EncodedStreamObject
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

def rip_images_from_pages(reader: PdfFileReader, page_count: int) -> List[dict]:
    # For now, let's put the pages as images into this list and return it
    output_list = list()
    for page_number in range(page_count):
        page: PageObject = reader.getPage(page_number)
        page_x_object: DictionaryObject = page['/Resources']['/XObject'].getObject()

        image_number = 0

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
                if DEBUG:
                    print(f'Attempting to load image {image_number} on page {page_number}.')
                image = {
                    'name': f'{page_number}_{image_number}{image_file_extension}',
                    'image_data': Image.frombytes(image_mode, (width, height), image_data) \
                        if image_mode == 'RGB' \
                            else Image.open(BytesIO(image_data))
                }
                output_list.append(image)
                image_number += 1
    return output_list


def save_image(image_metadata: dict) -> bool:
    image_file_name: str = image_metadata['name']
    image_data: Image = image_metadata['image_data']
    image_save_directory = image_metadata['image_save_directory']
    image_save_path = f'{image_save_directory}{image_file_name}'
    image_data.save(image_save_path)
    image_data.close()


def dump_images(output_directory: str, images: List[dict]) -> None:
    print(f'Dumping {len(images)} images...')
    image_save_directory = f'./{output_directory}/images/'
    os.makedirs(image_save_directory, exist_ok=True)

    progress = 0
    print('Saving images from PDF...')
    for image in images:
        image['image_save_directory'] = image_save_directory
    
    for image in images:
        save_image(image)
        progress += 1
        print(f'Saved image {progress} / {len(images)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The PDF to read.')
    parser.add_argument('-o', '--output-directory', required=False, default='scratch', help='Where to place the output of the script.')
    parser.add_argument('-d', '--dump-images', required=False, action='store_true', help='Output the images from the PDF.')
    args = parser.parse_args()

    input_file = args.input_file
    output_directory = args.output_directory

    reader = PdfFileReader(input_file, strict=DEBUG, warndest=sys.stderr if DEBUG else os.devnull)
    info = get_pdf_info(reader)
    page_count: int = info['page_count']
    document_images: List[dict] = rip_images_from_pages(reader, page_count)
    if args.dump_images:
        dump_images(output_directory, document_images)



if __name__ == '__main__':
    main()
