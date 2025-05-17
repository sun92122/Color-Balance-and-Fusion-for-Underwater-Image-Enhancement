import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_float(img):
    """
    Convert image to float format
    parameters
    ----------
    img: RGB image

    returns
    -------
    img: float image
    """
    return np.clip(img / 255.0, 0, 1)


def to_uint8(img):
    """
    Convert image to uint8 format

    parameters
    ----------
    img:
        RGB image

    returns
    -------
    img:
        uint8 image
    """
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_image(img, filename):
    """
    Save image to file

    parameters
    ----------
    img:
        RGB image
    filename:
        output filename (path)
    """
    temp_img = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, temp_img)
    print(f"Image saved as {filename}")


def push_image_to_buffer(image_buffer: list, img=None, title="Image", cmap=None):
    """
    Push image to buffer for later use

    parameters
    ----------
    image_buffer: list
        list to store images
    img:
        RGB image
    title:
        image title
    cmap:
        colormap (default None)
    """
    image_buffer.append((img, title, cmap))


def show_buffered_images(image_buffer: list, image_title="", save_path=None):
    """
    Show all images in the buffer, max 7 per row.
    The last image is displayed in the rightmost two columns, spanning all rows.

    parameters
    ----------
    image_buffer: list
        list to store images
    """
    n = len(image_buffer)
    if n == 0:
        return

    if n == 1:
        # Only one image, just show it
        img, title, cmap = image_buffer[0]
        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        image_buffer.clear()
        return

    cols = 7
    main_cols = cols - 2  # columns for normal images
    # Exclude last image, fill main_cols per row
    rows = (n - 1 + main_cols - 1) // main_cols

    fig = plt.figure(figsize=(cols * 3, rows * 3))

    # Show all except last image
    for idx in range(n - 1):
        row = idx // main_cols
        col = idx % main_cols
        ax = plt.subplot2grid((rows, cols), (row, col))
        img, title, cmap = image_buffer[idx]
        if img is not None:
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
        ax.axis("off")

    # Show last image, spanning all rows in the last two columns
    img, title, cmap = image_buffer[-1]
    ax = plt.subplot2grid((rows, cols), (0, cols - 2), rowspan=rows, colspan=2)
    if img is not None:
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
    ax.axis("off")

    plt.suptitle(image_title, fontsize=16)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Image saved as {save_path}")

    plt.show()
    image_buffer.clear()
