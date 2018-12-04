"""
The purpose of this script is slightly different than the others.
This does not aim to be a very comprehensive wrapper of opencv since
opencv has a very good python wrapper already. Rather, I write this
module mainly for myself since I am not yet familiar with the usage of opencv.
"""
import cv2 as opencv


def cartesian_to_polar(source_image, center, max_radius):
    """
    This function map an image over a cartesian grid to a polar grid.
    This function fill pixels outside the region with 0s.

    :param source_image: The source image.
    :param center: The center of the map in the cartesian coordinate
    :param max_radius: The maximum radius to map.
    :return: The image on a polar grid.
    """
    return opencv.linearPolar(src=source_image,
                              center=center,
                              maxRadius=max_radius,
                              flags=opencv.WARP_FILL_OUTLIERS)

