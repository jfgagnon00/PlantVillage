import cv2
import helpers as hlp

from helpers.visual_words.VisualWords import VisualWords


if __name__ == "__main__":
    configs = hlp.get_configs("config_overrides.json")
    visual_words = VisualWords.from_sift_configs(configs)
