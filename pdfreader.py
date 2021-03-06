import argparse
import attr
import cv2
import math
import numpy
import os
import pytesseract
import sys
import uuid

from io import BytesIO
from multiprocessing import cpu_count, Pool
from typing import Iterator, List, Tuple

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

@attr.s(auto_attribs=True)
class ImageData(object):
    name: str = attr.ib(kw_only=True)
    page: int = attr.ib(kw_only=True)
    image_number: int = attr.ib(kw_only=True)
    image: Image = attr.ib(kw_only=True)

def get_pdf_info(reader: PdfFileReader) -> dict:
    standard_info = reader.documentInfo
    return {
        'page_count': reader.getNumPages(),
        # The problem here is that I don't know what's standard and what isn't.
        'creation_date': standard_info.get('/CreationDate'),
        'page_layout': reader.getPageLayout(),
    }

def rip_images_from_pages(reader: PdfFileReader, page_count: int) -> Iterator[ImageData]:
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
                image_bytes: bytes = encoded_image._data
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
                image = ImageData(
                    name=f'{page_number}_{image_number}{image_file_extension}',
                    page=page_number,
                    image_number=image_number,
                    image=Image.frombytes(image_mode, (width, height), image_bytes) \
                        if image_mode == 'RGB' \
                            else Image.open(BytesIO(image_bytes))
                )

                image_number += 1
                # NOTE
                # Instead of returning all of the loaded images at once, we
                # are yielding them one at a time lazily. This is because Pillow
                # leaks memory like a sieve when you try and juggle images around.
                # No matter how careful you are to close your images as soon as you're
                # done messing around with them, it still runs up memory. Until this
                # gets fixed, we're forced to do it this way.
                yield image


def save_image(image_metadata: dict) -> bool:
    try:
        image_file_name: str = image_metadata['name']
        image_data: Image = image_metadata['image_data']
        image_save_directory = image_metadata['image_save_directory']
        image_save_path = f'{image_save_directory}{image_file_name}'
        image_data.save(image_save_path)
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

    # TODO: Maybe we can do houghlines and find the x
    # location of a line with +/- 0 degree skew?

    image_width, image_height = image.size
    left_page = image.crop(
        (0, 0, image_width // 2, image_height)
    )

    right_page = image.crop(
        (image_width // 2, 0, image_width, image_height)
    )

    return (left_page, right_page)


def split_page(document_image: ImageData) -> List[ImageData]:
    new_document_images_list = list()
    # TODO: This isn't very forward compatible. It breaks if there is
    # more than one image on a page. We'll come back to that if we need
    # to, but right now, I don't wanna get tied up in a whole data modeling
    # exercise for an MVP.
    page: int = document_image.page
    original_name: str = document_image.name
    image_number: int = document_image.image_number
    original_extension: str = original_name[-4:]  # NOTE: Dirty
    raw_image: Image = document_image.image
    left_page, right_page = do_page_split(raw_image)

    raw_image.close()
    new_document_images_list.append(
        ImageData(
            name=original_name,
            page=page,
            image_number=image_number,
            image=left_page
        )
    )

    new_document_images_list.append(
        ImageData(
            name=f'{page}_{image_number + 1}{original_extension}',
            page=page,
            image_number=image_number + 1,
            image=right_page    
        )
    )

    return new_document_images_list

def write_page_text(output_directory: str, page_data: ImageData, page_text: str) -> None:
    pages_directory = f'{output_directory}/pages'
    if not os.path.exists(pages_directory):
        os.makedirs(pages_directory, exist_ok=True)
    elif not os.path.isdir(pages_directory):
        print(f'{pages_directory} exists, but isn\'t a usable directory. Pick somewhere else.')
    
    page_number = page_data.page
    image_number = page_data.image_number
    with open(f'{pages_directory}/{page_number}_{image_number}.txt', 'w') as file_handle:
        file_handle.write(page_text)


# Hough line param constants
HOUGH_STEP_SIZE_RHO = 1
# Canny edge detection param constants
CANNY_FIRST_THRESHOLD = 50
CANNY_SECOND_THRESHOLD = 200
# The maximum angle of a line that we'll consider as a valid
# returned line from the Hough process
MAXIMUM_SKEW_ANGLE = 10
# How many times we try to get a valid houghlines result before bailing
HOUGH_ITERATION_TRIES = 12
def get_text_skew_angle(image: Image) -> float:
    quantified_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    edge_detected_image = cv2.Canny(quantified_image, CANNY_FIRST_THRESHOLD, CANNY_SECOND_THRESHOLD)

    minimum_line_size = 250
    found_line_configuration = False
    lines: numpy.ndarray = None
    iteration_count = 0
    while not found_line_configuration:
        lines = cv2.HoughLines(
            edge_detected_image,
            HOUGH_STEP_SIZE_RHO,
            math.pi / 180,
            # NOTE
            # This is the minimum size to constitute a line. The correct
            # value for this is hotly debated by my computer. If I get this
            # wrong, there are either
            # - 1 zillion incorrect lines
            # - no lines at all
            # So we're going to iterate until we find a good value for it.
            minimum_line_size        
        )
        if lines is None:
            lines = list()
        if len(lines) == 0:
            if DEBUG:
                print()
                # NOTE:
                # There's a careful balance here. We backpedal just enough that this
                # acts like a sliding window until we get some decent lines. If we ever
                # get caught in an infinite loop, we should probably increase the window
                # size to have more opportunities for valid line counts -- at the expense
                # of ( even worse ) performance.
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
        iteration_count += 1
        if iteration_count > HOUGH_ITERATION_TRIES:
            print('Failed to find usable lines for generating text skew angle. Continuing.')
            found_line_configuration = True
    potential_angles = list()
    for line in lines:
        rho,theta = line[0]
        angle_degrees = (theta * 180) / math.pi
        if DEBUG:
            print(f'theta = {theta} radians, or {angle_degrees} degrees')
        # Throw out serious outlier lines. If there's more than 10 degrees of skew, we're hosed.
        if abs(90 - angle_degrees) < MAXIMUM_SKEW_ANGLE:
            potential_angles.append(angle_degrees)

    if len(potential_angles) != 0:
        return 90 - sum(potential_angles) / len(potential_angles)
    else:
        # NOTE:
        # This isn't always an error case. Sometimes there's nothing / very little
        # on the page to go by, and this winds up acting as a filter.
        if DEBUG:
            canny_filename = f'canny-{uuid.uuid4()}.jpg'
            print(f'No potential angles found, writing edge-detected image to {canny_filename}')
            cv2.imwrite(canny_filename, edge_detected_image)
        return 0


MEDIAN_BLUR_FILTER_KERNEL_SIZE = 3
THRESHOLDING_CLASSIFIER = 45
THRESHOLDING_BLOCK_SIZE = 11 # Must be odd
THRESHOLDING_SUBTRACTED_NEIGHBORHOOD_CONSTANT = 4
def clean_image(image: Image) -> numpy.ndarray:
    # Median blur auto-detects kernel areas, takes medians of the pixels
    # in those areas, and replaces the pixels in the kernel areas with that value.
    filtered_grayscale_image = cv2.medianBlur(
        cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2GRAY), # Grayscale version of image
        MEDIAN_BLUR_FILTER_KERNEL_SIZE
    )

    # cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
    # C is a constant that is subtracted from the mean or weighted sum of the neighbourhood pixels.

    thresholded_image = cv2.adaptiveThreshold(
        filtered_grayscale_image,
        255, # Value assigned to pixels exceeding the threshold
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # adaptiveMethod
        cv2.THRESH_BINARY,  # thresholdType
        THRESHOLDING_BLOCK_SIZE, # Value used to classify pixel values
        THRESHOLDING_SUBTRACTED_NEIGHBORHOOD_CONSTANT
    )
    if DEBUG:
        cv2.imwrite(f'grayscale-{uuid.uuid4()}.jpg', thresholded_image)

    return thresholded_image

def preprocess_image(image_data: ImageData) -> numpy.ndarray:
    image: Image = image_data.image
    skew_angle = get_text_skew_angle(image)
    rotated_image = image.rotate(-skew_angle)
    return clean_image(rotated_image)

def get_and_save_text(preprocessed_image: numpy.ndarray, image_data: ImageData, output_directory: str) -> None:
        text: str = pytesseract.image_to_string(preprocessed_image, lang='srp') or str()
        write_page_text(output_directory, image_data, text)

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

    processing_pool = Pool(2)

    print('Handling pages...\n')
    pages_handled = 0
    for document_image in rip_images_from_pages(reader, page_count):
        if args.double_page:
            document_images: List[ImageData] = split_page(document_image)
        else:
            document_images = [document_image]

        if args.dump_images:
            dump_images(output_directory, document_images)
        
        # This is also the default case where it's one book page per
        # pdf page.
        left_image_data: ImageData = document_images[0]
        preprocessed_left_image: numpy.ndarray = preprocess_image(left_image_data)
        result_left = processing_pool.apply_async(get_and_save_text, (preprocessed_left_image, left_image_data, output_directory))

        if args.double_page:
            right_image_data = document_images[1]
            preprocessed_right_image: numpy.ndarray = preprocess_image(right_image_data)
            result_right = processing_pool.apply_async(get_and_save_text, (preprocessed_right_image, right_image_data, output_directory))

        # Await both here.
        _ = result_left.get()
        print(f'Wrote a new page to {left_image_data.page}_{left_image_data.image_number}.txt!\r', end='')
        if args.double_page:
            _ = result_right.get()
            print(f'Wrote a new page to {right_image_data.page}_{right_image_data.image_number}.txt!\r', end='')

        pages_handled += 1


if __name__ == '__main__':
    main()
