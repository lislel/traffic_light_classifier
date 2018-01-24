##Code provided by Udacity, lines 2-22
import cv2  # computer vision library

import random
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.image as mpimg # for loading in images

def load_dataset(image_dir):
    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green"]

    # Iterate through each color folder
    for im_type in image_types:
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            # Read in the image
            im = mpimg.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = load_dataset(IMAGE_DIR_TRAINING)

#Lines 22 through 58 written by me
##Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
image_index = 0
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]

for image in range(0,1187):
    if IMAGE_LIST[image][1] == "yellow":
        selected_image = IMAGE_LIST[image][0]

## Print out 1. The shape of the image and 2. The image's label
plt.imshow(selected_image)
print("Shape: "+str(selected_image.shape))
print("Label: " + str(selected_label))
# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[0][0]
plt.figure()
plt.imshow(selected_image)


def standardize_input(image):
    ## Resize image and pre-process so that all "standard" images are the same size
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    return standard_im


def one_hot_encode(label):
    ## Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = []
    if label == 'red':
        one_hot_encoded = [1, 0, 0]
    elif label == 'yellow':
        one_hot_encoded = [0, 1, 0]
    elif label == 'green':
        one_hot_encoded = [0, 0, 1]
    return one_hot_encoded

# The following code provided by Udactiy
import test_functions
tests = test_functions.Tests()


def standardize(image_list):
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list

## Code lines 84 through 152 written by me. Display a standardized image and its label
image_num = 0
selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

plt.imshow(selected_image)
print("Shape: "+str(selected_image.shape))
print("Label:" + str(selected_label))

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


def create_feature(test_im):
    hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

    # HSV channels
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Define our color selection boundaries
    lower = np.array([0])
    upper = np.array([60])
    # Define the masked area, using the saturation channel
    mask = cv2.inRange(s, lower, upper)
    # create a copy of the value channel
    mask_im = np.copy(v)

    # create two new images: mask the test image copy and the value copy
    mask_im[mask != 0] = [0]
    # create empty list of guess color, which is the guess the algorithm witll make, and bright pix, which will find the brightest pixel
    guess_color = []
    bright_pix = 0

    # iterate over masked v image, to find the brightest pix excluded from the mask, and store the indexes of that pixel
    for i in range(0, 32):
        for j in range(0, 32):
            if mask_im[i][j] >= bright_pix:
                bright_pix = mask_im[i][j]
                max_i = i
                max_j = j

    # find the corresponding hue value for that pixel using the h channel. Depending on the hue number, determine which color
    # the traffic light is and returng that color

    max_pix = h[max_i][max_j]

    if bright_pix > 0 and max_pix > 170:
        guess_color = [1, 0, 0]
    elif bright_pix > 0 and max_pix >= 0 and max_pix < 10:
        guess_color = [1, 0, 0]
    elif bright_pix > 0 and max_pix >= 70 and max_pix < 105:
        guess_color = [0, 0, 1]
    elif bright_pix > 0 and max_pix >= 10 and max_pix < 30:
        guess_color = [0, 1, 0]
    else:
        guess_color = [1, 0, 0]

    return (guess_color)


def estimate_label(rgb_image):
    ## Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = create_feature(rgb_image)

    return predicted_label

#The rest of the code is provided by Udacity
# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert (len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert (len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels
        if (predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            print("predicted=", predicted_label)
            print("true=", true_label)
            print("\n")

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct / total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))

for num in range(0,14):
    a = MISCLASSIFIED[num][0]
    b = create_feature(a)
    pred = MISCLASSIFIED[num][1]
    tru = MISCLASSIFIED[num][1]

    plt.figure()
    plt.imshow(a)
    print(num)
    print('predicted=', MISCLASSIFIED[num][1])
    print('true=', MISCLASSIFIED[num][2])

# Code provided by udacity - Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")
