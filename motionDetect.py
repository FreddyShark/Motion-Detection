"""
Acknowledgements to lab solutions provided by Dr Cai, Tom (University of Sydney) utilised as a guide
"""
import numpy
import cv2
import math
import os
import sys
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)
# check if directory exists for extracted motion detection fields
if not os.path.isdir(os.path.join(os.getcwd(), 'motionDetected')):
    os.mkdir("motionDetected")
else:
    print('stored motion detection will be overwritten.\n')

capture = cv2.VideoCapture('SNATCH_2sec.mp4')
if not capture.isOpened():
    print('video clip failed to open....Exiting.')
    sys.exit(1)

num_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# size of grid block dimensions x and y (5x5 pixels)
block_size = 5
block_rad = int(block_size/2)
num_of_blocks = int((frame_width*frame_height)/(block_size**2))
# size of radius of search field (in blocks)(5x5 block area centred at Bi+1)
search_rad = 2
pad_size = block_size*search_rad
# threshold for highlighting a pixel in boundary with large movement
dist_thresh = 2*block_size
# threshold for conversion of binary image to be set according to video
binary_thresh = 127


def insert_padding(img):
    """
    This method was adapted from week 2 lab solution
    :param img: image to pad
    :return: paddded image
    """
    padding_3_dims = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    # apply padding in the above dimensions with values 0
    img_padded = numpy.pad(img, padding_3_dims, 'constant', constant_values=0)
    return img_padded


def get_binary_image(image):
    """
    :param image: image to convert to binary
    :return: the image converted to inverse black and white
    """
    grey_image = image.copy()
    # e.g. [:,:,2] is all ranges in x,y dimensions and channel 2  (formula for grey scale taken from lab 3 solution)
    set_grey_img = 0.212671*grey_image[:, :, 2] + 0.715160*grey_image[:, :, 0] + 0.072169*grey_image[:, :, 1]
    # set RGB channels to grey value
    grey_image[:,:,0] = set_grey_img
    grey_image[:,:,1] = set_grey_img
    grey_image[:,:,2] = set_grey_img
    # get inverse binary image for boundary analysis
    # lab formula
    # matrix addition // (grey image (pixels < 127) = (1 or 0) + corresponding pos in new zero matrix )*(value for white)
    set_binary_image = ((grey_image < binary_thresh) + numpy.zeros(image.shape)) * 255
    # use Green channel binary values for Green screen demo video
    binary_image = set_binary_image[:, :, 1]
    cv2.imwrite('black_white.png', binary_image)
    return binary_image


def get_matching_block(frame1, frame2, center_y1, center_x1):
    """
    :param frame1: frame fi from video
    :param frame2: frame fi+1
    :param center_y1: center y coordinate of matching block
    :param center_x1: center x coordinate of matching block
    :return: center coordinates of closest matching block and the SSD to that block
    """
    # insignificant large numbers
    minSSD = 999999
    # for pythagoras to ensure closest block gets picked when equality occurs of SSD value
    min_d = 999999
    # this finds the center of the block to compare
    for factor_y in range(-search_rad, search_rad + 1):
        center_y2 = center_y1 + block_size*factor_y
        y_dist = center_y1 - abs(center_y2)
        for factor_x in range(-search_rad, search_rad + 1):
            center_x2 = center_x1 + block_size*factor_x
            x_dist = center_x1 - abs(center_x2)
            # pythagoras
            d = math.sqrt((y_dist**2 + x_dist**2))
            if d < min_d:
                min_d = d
            dist = 0
            # traversal of pixels in potential Bi+1 block
            # compare corresponding pixel positions with source block in f1 and neighbour block in f2
            y1 = center_y1 - block_rad      # start pos.
            for y2 in range(center_y2 - block_rad, (center_y2 - block_rad + block_size)):
                x1 = center_x1 - block_rad      # start pos
                for x2 in range(center_x2 - block_rad, (center_x2 - block_rad + block_size)):
                    try:
                        # displacement formula for RGB channels of each pixel in block
                        dist = dist + (frame1[y1][x1][0] - frame2[y2][x2][0]) ** 2 + (frame1[y1][x1][1] - frame2[y2][x2][1]) ** 2 + (frame1[y1][x1][2] - frame2[y2][x2][2]) ** 2
                    except RuntimeWarning:
                        pass
                    x1 += 1
                y1 += 1

            SSD = math.sqrt(dist)
            if SSD < minSSD:
                minSSD = SSD
                yf2 = center_y2
                xf2 = center_x2
            elif SSD == minSSD and d == min_d:
                yf2 = center_y2
                xf2 = center_x2
    # return centre of shortest RGB distance (smallest difference) block
    return yf2, xf2, minSSD


def check_if_boundary(image, y, x):
    """
    :param image: frame of video
    :param y: center y coordinate of area in question
    :param x: center x coordinate of area in question
    :return: Boolean (True if it is a boundary)
    """
    # define a kernel to cover grid block 5x5
    kernel = numpy.ones((block_size, block_size), numpy.uint8)
    # get component of image to compare
    image_section = image[y-block_rad:y+block_rad+1, x-block_rad:x+block_rad+1]
    test = kernel*image_section
    if not numpy.all(test) and numpy.any(test):
        return True
    else:
        return False


# iterate through frames
frame_count = 0
has_frames, frame = capture.read()

while has_frames:

    frame = insert_padding(frame)
    # holds vectors representing distance motion from point
    # vectors represented by end points x1, y1, x1', y1'
    vectors_y = numpy.zeros((num_of_blocks, 2))
    vectors_x = numpy.zeros((num_of_blocks, 2))
    # list to hold boolean values
    boundaries = [0]*num_of_blocks

    if frame_count >= 1:
        print('processing frame %d of %d\n....' % (frame_count, num_of_frames))
        # prepare binary for edge analysis
        binary_image = get_binary_image(prev_frame)
        # starting position of y (centre first block)
        y = pad_size + block_rad
        # for each block in frame
        index = 0
        while y < (frame_height + pad_size - block_rad):
            # starting position of x
            x = pad_size + block_rad
            while x < (frame_width + pad_size - block_rad):
                y2, x2, length = get_matching_block(prev_frame, frame, y, x)
                vectors_y[index] = (y, y2)
                vectors_x[index] = (x, x2)
                if length >= dist_thresh:
                    boundaries[index] = check_if_boundary(binary_image, y, x)
                index += 1
                # jump to next block center
                x += block_size
            # jump to next block center
            y += block_size
        for vec in range(index):
            img = cv2.arrowedLine(prev_frame, (int(vectors_x[vec][0]), int(vectors_y[vec][0])), (int(vectors_x[vec][1]), int(vectors_y[vec][1])), (100, 30, 150), 1, 4, 0, 0.3)
            if boundaries[vec]:
                cv2.line(prev_frame, (int(vectors_x[vec][0]), int(vectors_y[vec][0])), (int(vectors_x[vec][0]), int(vectors_y[vec][0])), (255, 0, 0), 4, 4)
        # remove padding for codec compatibility (resolution must be supported)
        save_frame = prev_frame[pad_size:frame_height+pad_size, pad_size:frame_width+pad_size, :]
        # save frames and number files.
        cv2.imwrite('motionDetected/frame%d.tif' % frame_count, save_frame)

    prev_frame = frame.copy()
    has_frames, frame = capture.read()
    frame_count += 1

print('frame processing completed.')
out = cv2.VideoWriter('vid_motion.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
frame_count = 1

while frame_count < num_of_frames:
    img = cv2.imread('motionDetected/frame%d.tif' % frame_count)
    out.write(img)
    frame_count += 1
capture.release()
