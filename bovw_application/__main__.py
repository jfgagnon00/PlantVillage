import cv2
import helpers as hlp

from helpers.visual_words.VisualWords import VisualWords


if __name__ == "__main__":
    configs = hlp.get_configs("config_overrides.json")
    visual_words = VisualWords.from_sift_configs(configs)

    # image = cv2.imread("dataset/demo/Grape___healthy/image (1).JPG")
    image = cv2.imread("dataset/demo/Grape___Esca_(Black_Measles)/image (45).JPG")

    key_points, features, tf_idf, label = visual_words.predict(image)

    print("Image shape:", image.shape)
    print("Key points:", len(key_points))
    print("Features shape:", features.shape)
    print("TF-IDF shape:", tf_idf.shape)
    print("Label:", label)
