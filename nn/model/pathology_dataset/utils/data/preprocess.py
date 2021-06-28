import shutil  # Provides high-level operations on files and collections of files
import os
import errno

# Define original image path, training (features + labels), and testing (features)
SRC_DIR = "../pathology_dataset/dataset/images"

# TRAIN_VAL_DIR = "/c/Users/nguye/OneDrive/Desktop/Junior/Project Git/aia_msa/pathology_dataset/dataset/working/train_val_images"
# TEST_DIR = "/c/Users/nguye/OneDrive/Desktop/Junior/Project Git/aia_msa/pathology_dataset/dataset/working/test_images"

TRAIN_VAL_DIR = "../pathology_dataset/dataset/working/train_val_images"
TEST_DIR = "../pathology_dataset/dataset/working/test_images"


def train_test_split():
    # Create folders
    if not os.path.isdir(TRAIN_VAL_DIR):
        os.mkdir(TRAIN_VAL_DIR)
    if not os.path.isdir(TEST_DIR):
        os.mkdir(TEST_DIR)

    # Put images in correct folder
    if len([f for f in os.listdir(TEST_DIR)]) == 0:
        all_images_names = os.listdir(SRC_DIR)
        for image in all_images_names:
            if "Train" in image:
                shutil.copy(SRC_DIR + "/" + image, TRAIN_VAL_DIR)
            elif "Test" in image:
                shutil.copy(SRC_DIR + "/" + image, TEST_DIR)
            else:
                print("Can't place image in Train or Test folder")

    # Check for errors
    total = len([file for file in os.listdir(SRC_DIR)])
    train_val_total = len([file for file in os.listdir(TRAIN_VAL_DIR)])
    test_total = len([file for file in os.listdir(TEST_DIR)])
    print(f"It is {total == train_val_total + test_total}")
    print(total)
    print(train_val_total)
    print(test_total)


if __name__ == "__main__":
    train_test_split()
