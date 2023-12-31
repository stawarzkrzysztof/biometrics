#  modules you gave us
import cv2
import os
import argparse
import random
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path
from collections import Counter

# AI modules
import torch
from facenet_pytorch import InceptionResnetV1

from sklearn.metrics import classification_report



#  global variables

#  resized images dimentions
IMG_HEIGHT = 160
IMG_WIDTH = 160

#  treshold to make sure there are at least this many photos of a person
N_FACES_THRESHOLD = 20

#  percentage of data divided into training
TRAIN_FRAC = .75


#  funtions


def read_labels(csv_path: str = None) -> Dict[int, int]:
    """ 
    Read labels from Caltech database .csv file and resturn as dict {id: label}

    Args:
        csv_path (str): Path to a CSV file.

    Returns:
        Dict[int, int]: Map of id:label pairs
    """
    assert csv_path is not None, "you didn't provide me any .csv file path!!"

    csv_path = Path(csv_path)

    assert csv_path.is_absolute(
    ), "Im sorry, but i don't know how to handle your relative path. Provide me with and absolute one and try again!"

    assert csv_path.exists(), "this path does not exist or may not be an absolute path to a file"

    mydict: Dict[int, int] = {}

    file: pd.core.frame.DataFrame = pd.read_csv(csv_path, header=None)

    for id, row in enumerate(file.values):
        label: int = row[0]
        mydict[id+1] = label

    return mydict


def read_ROIs(path: str = None) -> np.ndarray:
    """ 
    Reading ROI's array

    Args:
        path (str): Path to a .MAT file.

    Returns:
        np.ndarray: ROI's array
    """
    assert path is not None, "you didn't provide me any .mat file path!!"

    file: dict = loadmat(path)
    rois: np.ndarray = file['SubDir_Data']

    return rois


def read_img_paths(directory: str = None) -> List[str]:
    """
    Read image file names from given directory and return as a list of str: "dir/name.jpg"

    Args:
        dir (str): A relative directory to folder with images. Defaults to None.

    Returns:
        List[str]: A list with relative directories to images.
    """
    assert directory is not None, "you didn't provide me with any path!"

    dir = Path(directory)

    assert dir.exists(), "Your path does not exist! or may not be an absolute path to a file"

    assert dir.is_absolute(), "Im sorry, but i don't know how to handle your relative path. Provide me with and absolute one and try again!"

    script_dir = Path.cwd()
    directory_path = Path(dir)

    #  creating a list of a strings with paths to a images in this dir
    filename_list: List[str] = [
        f"./{str(file.relative_to(script_dir))}" for file in directory_path.glob('**/*') if str(file).endswith('.jpg')]

    assert all(path.endswith('.jpg')
               for path in filename_list), "your files are not all .jpg'!"

    #  making sure to sort them so i dont have to do anything with mixes indexes in many lists or arrays
    filename_list.sort()

    return filename_list


def img_preprocessing(path: str, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess images to use them as inputs to FRs

    Args:
        path (str): relative path to an image
        top_left (Tuple[int, int]): top left [x1, y1] coordinates of an image
        bottom_right (Tuple[int, int]): bottom_right [x2, y2] coordinates of an image

    Returns:
        np.ndarray: preprocessed image, cropped to its ROI, resized to 70x100 px and in grayscale color
    """
    
    #  1 importing img
    imported_img: np.ndarray = cv2.imread(path)

    #  2 doing it in grey
    #  grayscale_img = cv2.cvtColor(imported_img, cv2.COLOR_RGB2GRAY)

    #  3 cropping
    cropped_img = imported_img[top_left[1]:bottom_right[1],
                               top_left[0]:bottom_right[0]]

    #  4 resizing
    resized_img: np.ndarray = cv2.resize(cropped_img, (IMG_WIDTH, IMG_HEIGHT))
    
    #  converting to a tensor
    resized_img = resized_img / 255.0
    out_img = torch.from_numpy(resized_img).permute(2, 0, 1).float()

    return out_img


if __name__ == "__main__":

    #  parsing path to a file in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    random.seed(random.randint(1, 100)) # to check it few times while re-running
    #random.seed(13)

    #  Read labels
    csv_path = os.path.join(args.path, "caltech_labels.csv")
    labels = read_labels(csv_path)

    #  Read ROIs
    mat_path = os.path.join(args.path, "ImageData.mat")
    rois = read_ROIs("/Users/stawager/Studia/s4/biometrics/1/p1/caltech/ImageData.mat")

    #  Read filenames
    filenames = read_img_paths(args.path)

    #  Create train and test data
    train_img: List[np.ndarray] = []
    train_lbl: List[int] = []
    test_img: List[np.ndarray] = []
    test_lbl: List[int] = []

    #  counting how many times does a label shows in values in labels dict
    how_many_labels = Counter(labels.values())

    for img_path in filenames:

        #  since I sorted images in img list, index of an image is it's path index in filenames
        img_index: int = filenames.index(img_path)
        #  remember python starts indexing at 1:)
        img_label: int = labels[img_index+1]
        n_faces: int = how_many_labels[img_label]

        if n_faces >= N_FACES_THRESHOLD:
            # those indexes are correct due to README.txt ;)
            img_top_left_cords: Tuple[int, int] = (int(rois[2, img_index]),
                                                int(rois[3, img_index]))
            img_bottom_right_cords: Tuple[int, int] = (int(rois[6, img_index]),
                                                    int(rois[7, img_index]))
            img_input: torch.tensor = img_preprocessing(img_path,
                                                    img_top_left_cords,
                                                    img_bottom_right_cords)

            if random.random() <= TRAIN_FRAC:
                train_img.append(img_input)
                train_lbl.append(img_label)
            else:
                test_img.append(img_input)
                test_lbl.append(img_label)
        
    device = 'cpu'
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    aligned = train_img
    
    #  embedding trainset
    aligned_train = torch.stack(aligned).to(device)
    embeddings_train = resnet(aligned_train).detach().cpu().numpy()
    
    svm = cv2.ml.SVM_create()
        
    #  SVM parameters
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    
    #  training the model
    svm.train(embeddings_train, cv2.ml.ROW_SAMPLE, np.array(train_lbl))

    #  embedding testsetś
    aligned_test = test_img
    aligned_test = torch.stack(aligned_test).to(device)
    embeddings_test = resnet(aligned_test).detach().cpu().numpy()

    # Predict labels for test images using the trained SVM model
    _, result = svm.predict(embeddings_test)

    # Convert the result to a list of predicted labels
    predicted_labels = result.tolist()

    correct = 0
    for predicted, label in zip(predicted_labels, test_lbl):
        if predicted == label:
            correct += 1
            
    accuracy = correct / len(test_lbl)
    #print(f"Accuracy: {correct} / {len(test_lbl)} = {accuracy*100}%")
    print(classification_report(test_lbl, predicted_labels))
    
    # for method_name, method in FRs.items():

    #     #  training models
    #     method.train(train_img, np.array(train_lbl))

    #     assert len(test_img) == len(test_lbl), "something is not yes my dude"
    #     correct_n: int = 0
    #     for i in tqdm(range(len(test_lbl))):
    #         image: np.array = test_img[i]
    #         label: int = test_lbl[i]

    #         #  testing models
    #         predicted_lbl, confidence = method.predict(image)

    #         #  if the model predicts correctly, we add 1 to a counter of corrent guesses
    #         if predicted_lbl == label:
    #             correct_n += 1

    #     #  printing the results
    #     print("{} accuracy = {:.2f} ({} out of {})".format(method_name,
    #         correct_n / float(len(test_lbl)), correct_n, len(test_lbl)))
    

quit()
