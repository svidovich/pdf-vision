import argparse
import numpy
import os
import sys

from io import BytesIO
from multiprocessing import cpu_count, Manager, Pool
from typing import List, Tuple

from PIL import Image

from PyPDF2 import PdfFileReader
from PyPDF2.generic import DictionaryObject, EncodedStreamObject
from PyPDF2.pdf import PageObject

DEBUG = False
PARALLEL_IMAGE_HANDLING = False

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
    try:
        image_file_name: str = image_metadata['name']
        image_data: Image = image_metadata['image_data']
        image_save_directory = image_metadata['image_save_directory']
        image_save_path = f'{image_save_directory}{image_file_name}'
        image_data.save(image_save_path)
        image_data.close()
        return True
    except Exception as e:
        print(f'Failed to save image: {e}')
        return False


def dump_images(output_directory: str, images: List[dict]) -> None:
    print(f'Dumping {len(images)} images...')
    image_save_directory = f'./{output_directory}/images/'
    os.makedirs(image_save_directory, exist_ok=True)

    for image in images:
        image['image_save_directory'] = image_save_directory

    if PARALLEL_IMAGE_HANDLING:
        with Pool(cpu_count() // 2, maxtasksperchild=20) as processing_pool:
            print(f'Built processing pool with {cpu_count()} processes.')
            processing_pool.map(
                save_image,
                images
            )
    else:
        progress = 0
        for image in images:
            save_image(image)
            progress += 1
            print(f'Saved image {progress} / {len(images)}\r', end='')

def split_pages(image: Image) -> Tuple:
    """
    Very stupid split that cuts an image in half.
    """

    image_width, image_height = image.size
    print(image_width)
    print(image_height)
    print(dir(image))
    print(image.crop.__doc__)
    left_page = image.crop(
        (0, 0, image_width // 2, image_height)
    )

    right_page = image.crop(
        (image_width // 2, 0, image_width, image_height)
    )

    return (left_page, right_page)


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
    print('Done collecting images.')
    if args.dump_images:
        dump_images(output_directory, document_images)

    test_image: dict = document_images[20]
    image_data = test_image['image_data']
    split_pages(image_data)


if __name__ == '__main__':
    main()
