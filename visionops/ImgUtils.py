"""Image conversion, visualization, and border-removal helpers."""

import os
import random
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from . import SysUtils as sysu
from ._quiet import quiet_print

# Utility functions PIL <-> Numpy
def pil_to_numpy(pil_image: Image.Image, output_bgr: bool = False, add_grayscale_dim: bool = False) -> np.ndarray:
    """
    Convert a PIL Image to a NumPy array, optionally converting to BGR format for OpenCV.

    This function converts a PIL Image object into a NumPy array, handling
    grayscale and color images (including those with an alpha channel).
    It provides an option to convert the color channel order to BGR, which is
    commonly used in OpenCV.

    Args:
        pil_image: The input PIL Image object.
        add_grayscale_dim: If True and the image is grayscale, the output
            NumPy array will have shape (H, W, 1) instead of (H, W).
            Defaults to False.
        output_bgr: If True, the color channels of the output NumPy array will be
            reordered from RGB/RGBA to BGR/BGRA. This is useful for
            compatibility with OpenCV. Defaults to False.

    Returns:
        A NumPy array representing the image in HWC format (height, width, channels).
        -   For RGB/BGR images, the shape is (H, W, 3).
        -   For RGBA/BGRA images, the shape is (H, W, 4).
        -   For grayscale images, the shape is (H, W) or (H, W, 1)
            depending on the value of `add_grayscale_dim`.
        The data type of the array is uint8, with pixel values in the range [0, 255].
    """
    # Convert palette-based images to RGB
    if pil_image.mode == 'P':
        pil_image = pil_image.convert('RGB')

    # Convert PIL Image to NumPy array
    numpy_image = np.array(pil_image)

    # Add channel dimension for grayscale images (H, W) -> (H, W, 1)
    if add_grayscale_dim and numpy_image.ndim == 2:
        numpy_image = np.expand_dims(numpy_image, axis=-1)

    # Ensure dtype is uint8 (0-255 range)
    if numpy_image.dtype != np.uint8:
        numpy_image = numpy_image.astype(np.uint8)

    # Handle BGR conversion, including cases with alpha channel
    if output_bgr:
        if numpy_image.ndim == 3:
            if numpy_image.shape[2] == 3:  # RGB to BGR
                numpy_image = numpy_image[:, :, ::-1]
            elif numpy_image.shape[2] == 4:  # RGBA to BGRA
                numpy_image = numpy_image[:, :, [2, 1, 0, 3]]  # Reorder and keep alpha
    return numpy_image


def numpy_to_pil(numpy_image: np.ndarray, input_bgr: bool = False) -> Image.Image:
    """
    Convert a NumPy array to a PIL Image, optionally handling BGR input.

    This function converts a NumPy array representing an image to a PIL Image.
    It handles both grayscale and color (RGB/RGBA or BGR/BGRA) NumPy arrays.

    Args:
        numpy_image: Input NumPy array representing the image.
            -   For grayscale, the array should have shape (H, W).
            -   For color, the array should have shape (H, W, C), where C is
                the number of channels (3 for RGB/BGR, 4 for RGBA/BGRA).
            The data type should be uint8.
        input_bgr: A boolean indicating whether the input NumPy array is in BGR
            format (used by OpenCV). If True, the color channels will be
            converted to RGB or RGBA before creating the PIL Image.
            Defaults to False.

    Returns:
        A PIL Image object.
    """
    # Ensure the input is a NumPy array
    if not isinstance(numpy_image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # Ensure the input array has a valid data type
    if numpy_image.dtype != np.uint8:
        raise ValueError("Input NumPy array must have dtype uint8.")

    # Check the number of dimensions and channels.
    if numpy_image.ndim == 2:
        # Grayscale image
        mode = "L"  # "L" mode for grayscale images
        pil_image = Image.fromarray(numpy_image, mode=mode)
    elif numpy_image.ndim == 3:
        # Color image
        channels = numpy_image.shape[2]
        if channels not in {3, 4}:
            raise ValueError(f"Color arrays must have 3 or 4 channels, got {channels}.")
        if input_bgr:
            if channels == 3:
                numpy_image = numpy_image[:, :, ::-1]  # BGR to RGB
                mode = "RGB"
            else:
                numpy_image = numpy_image[:, :, [2, 1, 0, 3]]  # BGRA to RGBA
                mode = "RGBA"
        else:
            mode = "RGB" if channels == 3 else "RGBA"
        pil_image = Image.fromarray(numpy_image, mode=mode)
    else:
        raise ValueError(
            f"Unsupported number of dimensions: {numpy_image.ndim}. Must be 2 or 3."
        )
    return pil_image

# Displaying multiple images as a grid
def image_grid(images, labels = None, figsize = (10,10), invert_color_channels = False,
               display_in_notebook = True, title=None):
    """
    Returns a Matplotlib figure object of a grid of images.
    A square grid is used to populate the images.
    Rows and columns of this grid are auto-computed based on the square-root of the total number of images.

    :param images: Input array must be of the shape N x H x W x C
                    N = No. of images
                    H = No. of rows
                    W = No. of columns
                    C = No. of color channels for each image (RGB), monochrome images should have len(C)==1
    :type images: Numpy array / List of Numpy image
    :param labels: List of string labels to print as the title of each image in grid.
                    If not None, length must match the length of `images` array. Defaults to None
    :type labels: List or None, optional
    :param figsize: Tuple of integers, this is passed on to plt.figure internally.
                    Defaults to (10,10)
    :type figsize: tuple, optional
    :param invert_color_channels: Whether to flip BGR->RGB. Useful to correct images read from OpenCV (BGR).
                                    Defaults to True.
    :type invert_color_channels: bool, optional
    :param title: String that will be set as the title of the figure. Defaults to None.
    :type title: str or None, optional
    :return: Matplot figure object
    :rtype: figure
    """

    if labels is not None and len(images) != len(labels):
        raise ValueError("Label length does not match image list length.")

    num_imgs = len(images)
    n_rows = int(np.ceil(np.sqrt(num_imgs)))
    n_cols = n_rows

    figure = Figure(figsize=figsize)
    if title is not None:
        figure.suptitle(title, fontsize=16)

    for i in range(num_imgs):
        ax = figure.add_subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            ax.set_title(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        img = images[i]
        if invert_color_channels and len(img.shape) > 2:
            img = img[:, :, ::-1]

        ax.imshow(img, cmap='gray') # cmap is applied only for grayscale images, else ignored!

    if display_in_notebook:
        try:
            from IPython.display import display
            display(figure)
        except ImportError:
            quiet_print("IPython is unavailable; returning the figure without notebook display.")

    return figure

def image_grid_from_pathlist(file_paths, labels = None, count = 25, random_seed = None, figsize = (10,10),
                             display_in_notebook = True, title=None):
    """
    Returns a Matplotlib figure object of a grid of images from a list of image file paths.
    Internally uses OpenCV to read the images. A square grid is used to populate the images.
    Rows and columns of this grid are auto-computed based on the square-root of the total number of images.

    :param file_paths: List of valid file paths to read the images from.
    :type file_paths: List of paths / path strings
    :param labels: List of string labels to print as the title of each image in grid.
                    If None, the image file names are used automatically as labels.
                    If not None, length must match the length of `file_paths`. Defaults to None
    :type labels: List or None, optional
    :param count: Max images to display. If this value is more than length of `file_paths`, latter is used.
                    Defaults to 25
    :type count: int, optional
    :param random_seed: To ransomize the order of images. Use None to skip shuffling. Defaults to None
    :type random_seed: Int or None, optional
    :param figsize: Tuple of integers, this is passed on to plt.figure internally.
                    Defaults to (10,10)
    :type figsize: tuple, optional
    :return: Matplot figure object
    :rtype: figure
    """
    count = len(file_paths) if count is None else min(max(int(count), 0), len(file_paths))
    labels = [os.path.basename(elem) for elem in file_paths] if labels is None else labels

    if len(file_paths) != len(labels):
        raise ValueError("Label length does not match file path list.")

    if random_seed is not None:
        rng = random.Random(random_seed)
        random_indices = rng.sample(range(len(file_paths)), count)
        temp_file_paths = []
        temp_labels = []
        for idx in random_indices:
            temp_file_paths.append(file_paths[idx])
            temp_labels.append(labels[idx])

        file_paths = temp_file_paths
        labels = temp_labels
    else:
        file_paths = file_paths[:count]
        labels = labels[:count]
    images = []
    for img_path in file_paths:
        with Image.open(img_path) as image:
            images.append(pil_to_numpy(image))

    return image_grid(images, labels, figsize, invert_color_channels=False,
                      display_in_notebook = display_in_notebook, title=title)


def image_grid_from_dir(root_dir, labels = None, count = 25, random_seed = None,
                        figsize = (10,10), image_extensions=("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP", "*.tiff", "*.TIFF", "*.tif", "*.TIF"),
                        display_in_notebook = True, title=None):
    """
    Returns a Matplotlib figure object of a grid of images from a list of all images found in the `root_dir` directory.
    Images are searched recursively, file extensions can be controlled with the `image_extensions` argument.
    Internally uses OpenCV to read these images. A square grid is used to populate the images.
    Rows and columns of this grid are auto-computed based on the square-root of the total number of images.

    :param root_dir: The directory to scan for image files
    :type root_dir: Path or str
    :param labels: List of string labels to print as the title of each image in grid.
                    If None, the image file names are used automatically as labels.
                    If not None, length must match the length of all image files found in `root_dir`.
                    Defaults to None
    :type labels: List or None, optional
    :param count: Max images to display. If this value is more than length of `file_paths`, latter is used.
                    Defaults to 25
    :type count: int, optional
    :param random_seed: To ransomize the order of images. Use None to skip shuffling. Defaults to None
    :type random_seed: Int or None, optional
    :param figsize: Tuple of integers, this is passed on to plt.figure internally.
                    Defaults to (10,10)
    :type figsize: tuple, optional
    :param image_extensions: List of image file extensions to search for.
                                Defaults to ["*.jpg", "*.JPG", "*.jpeg", ".JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP","*.tiff", "*.TIFF", "*.tif", "*.TIF"]
    :type image_extensions: list, optional
    :return: Matplot figure object
    :rtype: figure
    """

    file_paths = sysu.search(search_in_dirs=root_dir, patterns=image_extensions)
    return image_grid_from_pathlist(file_paths, labels, count, random_seed, figsize,
                                    display_in_notebook = display_in_notebook, title=title)


def image_grid_from_dataframe(df, path_col_name="path", label_col_name=None, count=25, random_seed = None, figsize = (10,10),
                              display_in_notebook = True, title=None):
    """
    Returns a Matplotlib figure object of a grid of images from a Pandas Dataframe `df` and name of path column `path_col_name`.
    Useful for viewing selec images, the dataframe can be filtered based on the conditions.
    Images are searched recursively, file extensions can be controlled with the `image_extensions` argument.
    Internally uses OpenCV to read these images. A square grid is used to populate the images.
    Rows and columns of this grid are auto-computed based on the square-root of the total number of images.

    :param df: Pandas Dataframe which has at least one column that contains the path to images
    :type df: pandas.DataFrame
    :param path_col_name: Name of the column that has path to images, defaults to "path"
    :type path_col_name: str, optional
    :param label_col_name: Name of the column that has labels for the images.
                            If None, the filenames from path are used as labels.
                            Defaults to None.
    :type label_col_name: str, optional
    :param count: Max images to display. If this value is more than length of `file_paths`, latter is used.
                    Defaults to 25
    :type count: int, optional
    :param random_seed: To ransomize the order of images. Use None to skip shuffling. Defaults to None
    :type random_seed: Int or None, optional
    :param figsize: Tuple of integers, this is passed on to plt.figure internally.
                    Defaults to (10,10)
    :type figsize: tuple, optional
    :return: Matplot figure object
    :rtype: figure
    """

    if path_col_name not in df.columns:
        raise ValueError(f"{path_col_name} does not exist in dataframe columns")
    if label_col_name: # Not None
        if label_col_name not in df.columns:
            raise ValueError(f"{label_col_name} does not exist in dataframe columns")
        labels = df[label_col_name].to_list()
    else:
        labels = None

    path_list = df[path_col_name].to_list()

    return image_grid_from_pathlist(path_list, labels, count, random_seed, figsize,
                                    display_in_notebook = display_in_notebook, title=title)



# Bounding-box ROI functions

def is_bordered(image, threshold: int = 20, dark_border: bool = True) -> bool:
    """
    Checks if a PIL / NumPy image has a uniform color border on any of the four
    sides, based on a given threshold and darkness preference.

    The function converts the image to grayscale and then examines the
    pixels along the left, right, top, and bottom edges. It determines
    if all pixels on each border are either darker than the specified
    threshold (if dark_border is True) or lighter than (255 - threshold)
    (if dark_border is False).

    Args:
        image: A PIL Image object or equivalent NumPy array.
        threshold: An integer threshold value (0-255 for 8-bit images)
                   used to determine if a pixel is considered dark or light.
                   Defaults to 20.
        dark_border: A boolean. If True (default), the function checks if
                     all border pixels have a grayscale value strictly less
                     than the threshold. If False, it checks if all border
                     pixels have a grayscale value strictly greater than
                     (255 - threshold).

    Returns:
        bool: True if all four borders satisfy the specified color condition,
              False otherwise.
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2] if img_array.ndim >= 2 else (0, 0)

    # Convert to grayscale if it's a color image
    if len(img_array.shape) == 3:
        gray_img = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray_img = img_array

    results = {'left': False, 'right': False, 'top': False, 'bottom': False}

    if gray_img.size == 0:
        return False

    if height == 1 and width == 1:
        pixel_value = gray_img[0, 0]
        if dark_border:
            condition_met = pixel_value < threshold
        else:
            condition_met = pixel_value > (255 - threshold)
        return condition_met

    light_threshold = 255 - threshold

    # Check top border
    if height > 0:
        top_border = gray_img[0, :]
        if dark_border:
            results['top'] = np.all(top_border < threshold)
        else:
            results['top'] = np.all(top_border > light_threshold)

    # Check bottom border
    if height > 1:
        bottom_border = gray_img[height - 1, :]
        if dark_border:
            results['bottom'] = np.all(bottom_border < threshold)
        elif not dark_border:
            results['bottom'] = np.all(bottom_border > light_threshold)
    elif height == 1:
        results['bottom'] = results['top']

    # Check left border
    if width > 0:
        left_border = gray_img[:, 0]
        if dark_border:
            results['left'] = np.all(left_border < threshold)
        elif not dark_border:
            results['left'] = np.all(left_border > light_threshold)

    # Check right border
    if width > 1:
        right_border = gray_img[:, width - 1]
        if dark_border:
            results['right'] = np.all(right_border < threshold)
        elif not dark_border:
            results['right'] = np.all(right_border > light_threshold)
    elif width == 1:
        results['right'] = results['left']

    return any(results.values())

def guess_border_color(image: Image.Image, threshold: int = 20) -> str:
    """
    Guesses the dominant color of an image's border.

    This function uses the `is_bordered` function to determine if the image
    has a uniform color border, and if so, whether that color is predominantly
    'dark' or 'light'.

    Args:
        image: A PIL Image object or equivalent NumPy array.
        threshold: The threshold value used by the `is_bordered` function
            to determine if a pixel is considered dark or light.
            Defaults to 20.

    Returns:
        str:  Returns "dark" if the border is uniformly dark, "light" if the
              border is uniformly light, and "" (empty string) if the border
              is not uniform or the image is empty.
    """
    is_black_border = is_bordered(image, threshold, dark_border=True)
    is_white_border = is_bordered(image, threshold, dark_border=False)

    if is_black_border == is_white_border:
        return ""
    return "dark" if is_black_border else "light"

def get_roi_bbox(image, threshold: int = 20, border_color: str = "dark") -> tuple:
    """
    Calculates the bounding box of the region of interest (ROI) within an image.

    The function identifies the ROI based on whether the image has a predominantly
    dark or light border.  It thresholds pixel values to distinguish the ROI
    from the background.

    Args:
        image: A PIL Image object or equivalent NumPy array
        threshold: Pixel value threshold (0-255) to differentiate ROI
            pixels from the background.  Defaults to 20.
        border_color:  Specifies the expected border color:
            - "dark":  Finds the bounding box of the non-dark region.
            - "light": Finds the bounding box of the non-light region.
            - Any other value: Returns the bounding box of the entire image.
            Defaults to "dark".

    Returns:
        tuple:  A tuple (x, y, width, height) representing the bounding box
                of the ROI.
                - Returns (0, 0, width, height) if `border_color` is not
                  "dark" or "light".
                - Returns None if no ROI is found (i.e., if the image
                  is empty or if all pixels are considered background
                  based on the threshold and `border_color`).
    """
    img_array = np.array(image)
    if img_array.size == 0:
        return None  # Return None for empty images

    height, width = img_array.shape[:2] if img_array.ndim >= 2 else (0, 0)

    if border_color not in {"dark", "light"}:
        return 0, 0, width, height

    if img_array.ndim == 3 and img_array.shape[2] == 1:
        img_array = img_array.squeeze(axis=2)
    if img_array.ndim == 2:
        channels = img_array
    elif img_array.ndim == 3:
        channels = img_array[..., :3]
    else:
        raise ValueError(f"Unsupported image dimensions: {img_array.shape}")

    if border_color == "dark":
        comparison = channels > threshold
    else:
        comparison = channels < (255 - threshold)
    mask = comparison if comparison.ndim == 2 else comparison.any(axis=2)

    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None

    rmin, cmin = coordinates.min(axis=0)
    rmax, cmax = coordinates.max(axis=0)
    rmin = max(int(rmin) - 1, 0)
    rmax = min(int(rmax) + 1, height - 1)
    cmin = max(int(cmin) - 1, 0)
    cmax = min(int(cmax) + 1, width - 1)
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1

def remove_border(image, threshold: int = 20, output_as_pil = True) -> Image.Image:
    """
    Crops an image to remove its uniform-color border, returning the region of interest (ROI).

    This function identifies the ROI by detecting the predominant border color
    (dark or light) and then calculates the bounding box that tightly encloses
    the non-border region.

    Args:
        image: A PIL Image object or equivalent NumPy array.
        threshold:  Pixel value threshold (0-255) used to distinguish border
            pixels from the ROI.  Defaults to 20.

    Returns:
        Image.Image: The cropped PIL Image containing the ROI.
            Returns the original image if no uniform color border is found
            or if the image is empty.
    """
    img_array = np.array(image)

    # Get the bounding box for the ROI
    border_color = guess_border_color(img_array, threshold)
    bbox = get_roi_bbox(img_array, threshold, border_color)

    if bbox is None:
        if output_as_pil:
            return numpy_to_pil(img_array)  # Return original image if no ROI found
        return img_array

    # Unpack bounding box coordinates
    x, y, w, h = bbox

    # Crop the image using the bounding box
    if img_array.ndim==2:   # 2D grayscale image
        img_array = img_array[y:y+h, x:x+w]
    else:
        img_array = img_array[y:y+h, x:x+w, :]

    if output_as_pil:
        return numpy_to_pil(img_array)
    return img_array
# Albumentations custom transform for remove_border

try:
    from albumentations.core.transforms_interface import ImageOnlyTransform

    class RemoveBorder(ImageOnlyTransform):
        """Albumentations custom transform for cropping to ROI with dark / light background.
        Usage:
        >>> import albumentations as A
        >>> from PIL import Image
        >>> import numpy as np
        >>> img_pil = Image.open("test.jpg")
        >>> img_arr = np.array(img_pil)
        >>> tfms = A.Compose([RemoveBorder()])
        >>> augmented = tfms(image = img_arr)
        >>> img_aug = augmented["image"]    # Numpy image
        """

        def __init__(self, threshold=20, always_apply=False, p=1.0):
            """Initialize border-removal threshold and transform probability."""
            super().__init__(always_apply, p)
            self.threshold = threshold

        def apply(self, img, **params):
            """
            Args:
                img (numpy.ndarray): Input image in HWC format (RGB or grayscale)

            Returns:
                numpy.ndarray: Cropped image
            """
            return remove_border(img, self.threshold, output_as_pil = False)

        def get_transform_init_args_names(self):
            """Return constructor argument names serialized by Albumentations."""
            return ("threshold",)

except ModuleNotFoundError:
    quiet_print(
        "WARNING: Can't import `albumentations.core.transforms_interface.ImageOnlyTransform` "
        "`RemoveBorder` transform will not be available"
    )
