import argparse
import cv2
import math
import numpy
import os
import sys
import uuid

from io import BytesIO
from multiprocessing import cpu_count, Pool
from typing import List, Tuple, Generator

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


def rip_images_from_pages(reader: PdfFileReader, page_count: int) -> Generator:
    # For now, let's put the pages as images into this list and return it
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
                    'page': page_number,
                    'image_number': image_number,
                    'image_data': Image.frombytes(image_mode, (width, height), image_data) \
                        if image_mode == 'RGB' \
                            else Image.open(BytesIO(image_data))
                }
                image_number += 1
                yield image


def save_image(image_metadata: dict) -> bool:
    try:
        image_file_name: str = image_metadata['name']
        image_data: Image = image_metadata['image_data']
        image_save_directory = image_metadata['image_save_directory']
        image_save_path = f'{image_save_directory}{image_file_name}'
        image_data.save(image_save_path)
        # image_data.close()
        return True
    except Exception as e:
        print(f'Failed to save image: {e}')
        return False


def dump_images(output_directory: str, images: List[dict]) -> None:
    image_save_directory = f'./{output_directory}/images/'
    os.makedirs(image_save_directory, exist_ok=True)

    for image in images:
        image['image_save_directory'] = image_save_directory
        save_image(image)


def do_page_split(image: Image) -> Tuple:
    """
    Very stupid split that cuts an image in half.
    """

    image_width, image_height = image.size
    left_page = image.crop(
        (0, 0, image_width // 2, image_height)
    )

    right_page = image.crop(
        (image_width // 2, 0, image_width, image_height)
    )

    return (left_page, right_page)


def split_page(document_image: List[dict]) -> List[dict]:
    new_document_images_list = list()
    # TODO: This isn't very forward compatible. It breaks if there is
    # more than one image on a page. We'll come back to that if we need
    # to, but right now, I don't wanna get tied up in a whole data modeling
    # exercise for an MVP.
    page: int = document_image['page']
    original_name: str = document_image['name']
    image_number: int = document_image['image_number']
    original_extension: str = original_name[-4:]  # NOTE: Dirty
    image_data = document_image['image_data']
    left_page, right_page = do_page_split(image_data)

    image_data.close()
    new_document_images_list.append(
        {
            'name': original_name,
            'page': page,
            'image_number': image_number,
            'image_data': left_page
        }
    )

    new_document_images_list.append(
        {
            'name': f'{page}_{image_number + 1}{original_extension}',
            'page': page,
            'image_number': image_number + 1,
            'image_data': right_page
        }
    )

    return new_document_images_list


# Hough line param constants
HOUGH_STEP_SIZE_RHO = 1
# Canny edge detection param constants
CANNY_FIRST_THRESHOLD = 50
CANNY_SECOND_THRESHOLD = 200

def get_text_skew_angle(image: Image) -> float:
    image_width, image_height = image.size
    quantified_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    edge_detected_image = cv2.Canny(quantified_image, CANNY_FIRST_THRESHOLD, CANNY_SECOND_THRESHOLD)

    minimum_line_size = 250
    found_line_configuration = False
    lines: numpy.ndarray = None
    while not found_line_configuration:
        lines = cv2.HoughLines(
            edge_detected_image,
            HOUGH_STEP_SIZE_RHO,
            math.pi / 180,
            # This is the minimum size to constitute a line. It's probably
            # a function of the image size or something, right?
            # image_width // 5
            minimum_line_size        
        )
        if lines is None:
            lines = list()
        if len(lines) == 0:
            if DEBUG:
                print()
                print(f'No lines found! Backpedaling line size threshold change.')
            minimum_line_size -= 4
        elif len(lines) > 5:
            minimum_line_size += 5
            if DEBUG:
                print()
                print(f'\tUnreliable line count {len(lines)}; trying again with minimum size {minimum_line_size}.')
            continue
        else:
            found_line_configuration = True
    potential_angles = list()
    for line in lines:
        rho,theta = line[0]
        angle_degrees = (theta * 180) / math.pi
        if DEBUG:
            print(f'theta = {theta} radians, or {angle_degrees} degrees')
        # Throw out serious outlier lines. If there's more than 10 degrees of skew, we're hosed.
        # TODO: This should be a constant.
        if abs(90 - angle_degrees) < 10:
            potential_angles.append(angle_degrees)

    if len(potential_angles) != 0:
        return 90 - sum(potential_angles) / len(potential_angles)
    else:
        if DEBUG:
            canny_filename = f'canny-{uuid.uuid4()}.jpg'
            print(f'No potential angles found, writing edge-detected image to {canny_filename}')
            cv2.imwrite(canny_filename, edge_detected_image)
        return 0


MEDIAN_BLUR_FILTER_KERNEL_SIZE = 3
def clean_image(image: Image) -> Image:
    # Median blur auto-detects kernel areas, takes medians of the pixels
    # in those areas, and replaces the pixels in the kernel areas with that value.
    filtered_grayscale_image = cv2.medianBlur(
        cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2GRAY), # Grayscale version of image
        MEDIAN_BLUR_FILTER_KERNEL_SIZE
    )

    cv2.imwrite(f'grayscale-{uuid.uuid4()}.jpg', filtered_grayscale_image)

    return filtered_grayscale_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The PDF to read.')
    parser.add_argument('-o', '--output-directory', required=False, default='scratch', help='Where to place the output of the script.')
    parser.add_argument('-d', '--dump-images', required=False, action='store_true', help='Output the images from the PDF.')
    parser.add_argument('-t', '--double-page', required=False, action='store_true', help='The PDF is double-page in layout ( each page of the PDF is two book pages )')
    parser.add_argument('-p', '--title-page-count', required=False, type=int, default=0, help='In a double-page spread, the number of single-page title images that exist in the PDF.')
    args = parser.parse_args()

    input_file = args.input_file
    output_directory = args.output_directory

    print(f'Reading data from {input_file}!')
    reader = PdfFileReader(input_file, strict=DEBUG, warndest=sys.stderr if DEBUG else os.devnull)
    info = get_pdf_info(reader)
    page_count: int = info['page_count']
    print(f'Identified {page_count} pages.')
    document_images: List[dict] = rip_images_from_pages(reader, page_count)

    print('Handling pages...\n')
    pages_handled = 0
    for document_image in rip_images_from_pages(reader, page_count):
        if args.double_page:
            document_images: List[dict] = split_page(document_image)
        else:
            document_images = [document_image]

        if args.dump_images:
            dump_images(output_directory, document_images)
        
        # if pages_handled == 45:
        left_image = document_images[0]['image_data']
        left_skew_angle = get_text_skew_angle(left_image)

        right_image = document_images[1]['image_data']
        right_skew_angle = get_text_skew_angle(right_image)
        if pages_handled == 9:
            rotated_left_image: Image = left_image.rotate(-left_skew_angle)
            # left_image.save('l_unrotated.jpg')
            # rotated_left_image.save('l_rotated.jpg')

            rotated_right_image: Image = right_image.rotate(-right_skew_angle)
            # right_image.save('r_unrotated.jpg')
            # rotated_right_image.save('r_rotated.jpg')

            cleaned_left_image: Image = clean_image(rotated_left_image)
            cleaned_right_image: Image = clean_image(rotated_right_image)
            exit(0)

        pages_handled += 1
        print(f'Page {pages_handled} skew angles are L {left_skew_angle} and R {right_skew_angle}')
        print(f'Done handling {pages_handled} pages.\r', end='')

if __name__ == '__main__':
    main()
