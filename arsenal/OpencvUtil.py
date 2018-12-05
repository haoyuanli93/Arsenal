"""
The purpose of this script is slightly different than the others.
This does not aim to be a very comprehensive wrapper of opencv since
opencv has a very good python wrapper already. Rather, I write this
module mainly for myself since I am not yet familiar with the usage of opencv.
"""
import cv2 as opencv


def cartesian_to_polar(cartesian_image, center, max_radius):
    """
    This function map an image over a cartesian grid to a polar grid.
    This function fill pixels outside the region with 0s.

    :param cartesian_image: The source image.
    :param center: The center of the map in the cartesian coordinate
    :param max_radius: The maximum radius to map.
    :return: The image on a polar grid.
    """
    return opencv.linearPolar(src=cartesian_image,
                              center=center,
                              maxRadius=max_radius,
                              flags=opencv.WARP_FILL_OUTLIERS)


def cartesian_to_polar_quick(cartesian_image, polar_image, center, max_radius):
    """
    This function map an image over a cartesian grid to a polar grid.
    This function fill pixels outside the region with 0s.

    :param cartesian_image: The source image.
    :param center: The center of the map in the cartesian coordinate
    :param max_radius: The maximum radius to map.
    :param polar_image: The variable holding the image on the polar grid.
    :return: The image on a polar grid.
    """
    return opencv.linearPolar(src=cartesian_image,
                              dst=polar_image,
                              center=center,
                              maxRadius=max_radius,
                              flags=opencv.WARP_FILL_OUTLIERS)


def polar_to_cartesian(polar_image, center, max_radius):
    """
    This function map an image over a cartesian grid to a polar grid.
    This function fill pixels outside the region with 0s.

    :param polar_image: The source image.
    :param center: The center of the map in the cartesian coordinate
    :param max_radius: The maximum radius to map.
    :return: The image on a polar grid.
    """
    return opencv.linearPolar(src=polar_image,
                              center=center,
                              maxRadius=max_radius,
                              flags=[opencv.WARP_FILL_OUTLIERS, opencv.WARP_INVERSE_MAP])


def polar_to_cartesian_quick(polar_image, cartesian_image, center, max_radius):
    """
    This function map an image over a cartesian grid to a polar grid.
    This function fill pixels outside the region with 0s.

    :param polar_image: The source image.
    :param center: The center of the map in the cartesian coordinate
    :param max_radius: The maximum radius to map.
    :param cartesian_image: The variable hold the image on the cartesian grid.
    :return: The image on a polar grid.
    """
    return opencv.linearPolar(src=polar_image,
                              center=center,
                              dst=cartesian_image,
                              maxRadius=max_radius,
                              flags=[opencv.WARP_FILL_OUTLIERS, opencv.WARP_INVERSE_MAP])
