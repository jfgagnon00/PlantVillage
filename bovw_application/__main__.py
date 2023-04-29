import cv2
import helpers as hlp

from helpers.visual_words.VisualWords import VisualWords


if __name__ == "__main__":
    # relire le modele pre-entraine
    configs = hlp.get_configs("config_overrides.json")
    visual_words = VisualWords.from_sift_configs(configs)

    # obtenir une image
    # image = cv2.imread("dataset/demo/Grape___healthy/image (1).JPG")
    # image = cv2.imread("dataset/demo/Grape___Esca_(Black_Measles)/image (45).JPG")
    image = cv2.imread("dataset/demo/astronaut.jpg")

    if not image is None:
        # s'assurer que l'image est de la meme taille que celles de lors de l'entrainement
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        # faire prediciton
        key_points, features, tf_idf, label = visual_words.predict(image)

        # afficher les resultats
        print("Image shape:", image.shape)
        print("Key points:", len(key_points))
        print("Features shape:", features.shape)
        print("TF-IDF shape:", tf_idf.shape)
        print("Label:", label)
