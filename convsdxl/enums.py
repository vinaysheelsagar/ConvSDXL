"""Module for defining enumerations used in the ConvSDXL project.

This module contains various enum classes that represent different types
of categorizations and constants that are used throughout the ConvSDXL
project.
"""


from enum import Enum


class DesignType(Enum):
    """Enum representing different design types for prompts.

    Attributes:
        DIGITAL_ART: Represents digital art design.
        ANIME: Represents anime style design.
        NEONPUNK: Represents neonpunk style design.
        PIXEL_ART: Represents pixel art design.
        MINIMALIST: Represents minimalist style design.
    """

    DIGITAL_ART = "digital_art"
    ANIME = "anime"
    NEONPUNK = "neonpunk"
    PIXEL_ART = "pixel_art"
    MINIMALIST = "minimalist"
