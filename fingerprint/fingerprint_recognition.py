import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import List, Tuple
from pathlib import Path


def read_img_paths(directory: str = None) -> List[str]:
    assert directory is not None, "you didn't provide me with any path!"

    dir = Path(directory)

    assert dir.exists(), "Your path does not exist! or may not be an absolute path to a file"

    assert dir.is_absolute(), "Im sorry, but i don't know how to handle your relative path. Provide me with and absolute one and try again!"

    script_dir = Path.cwd()
    directory_path = Path(dir)

    #  creating a list of a strings with paths to a images in this dir
    filename_list: List[str] = [
        f"./{str(file.relative_to(script_dir))}" for file in directory_path.glob('**/*') if str(file).endswith('.tif')]

    assert all(path.endswith('.tif')
               for path in filename_list), "your files are not all .tif'!"

    #  making sure to sort them so i dont have to do anything with mixes indexes in many lists or arrays
    filename_list.sort()

    return filename_list


def img_preprocessing(path: str = None) -> np.ndarray:
    """
    Read image from file and convert to grayscale
    """

    assert path is not None, "you didnt give me any path!"

    imported_img: np.ndarray = cv2.imread(path)
    gray: np.ndarray = cv2.cvtColor(imported_img, cv2.COLOR_RGB2GRAY)

    return gray


def get_features(gray: np.ndarray, alg_type: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute keypoints locations and their descriptors
    """
    assert alg_type == "sift" or alg_type == "orb", "i dont know your algotypeman"
    kp: List[np.ndarray] = []
    des: List[np.ndarray] = []

    object = cv2.SIFT.create() if alg_type == "sift" \
        else cv2.ORB.create()

    kp, des = object.detectAndCompute(gray, None)
        
    return kp, des


def match_fingerprints(features2match: np.ndarray, 
                       features_base: np.ndarray, 
                       alg_type: str, 
                       m_coeff=0.75) -> Tuple[List[str], List[str]]:
    """
    Match features and gather stats
    """
    
    assert alg_type == "sift" or alg_type == "orb", "i dont know the algo type dude!"
    y_pred: List[str] = list()
    y_test: List[str] = list()

    dm = cv2.BFMatcher.create(cv2.NORM_L2) if alg_type == "sift" \
    else cv2.BFMatcher.create(cv2.NORM_HAMMING)
    
    for finger_id in range(101, 111):
        
        for sample in features2match[str(finger_id)][alg_type]:
            
            sample_id = 101
            storing_samples = dict()
            for base_sample in [dic[alg_type][0] for dic in features_base.values()]:
                
                
                matches = dm.knnMatch(sample.astype(np.float32)
                                      if alg_type=="sift"
                                      else sample.astype(np.uint8), 
                                      base_sample.astype(np.float32)
                                      if alg_type=="sift"
                                      else base_sample.astype(np.uint8), 
                                      k=2)
                
                good_matches_num = 0
                for m, n in matches:
                    if m.distance < m_coeff * n.distance: 
                        good_matches_num += 1
                    
                storing_samples[sample_id] = good_matches_num
                sample_id += 1
                
            y_pred.append(max(storing_samples, key=storing_samples.get))
            y_test.append(finger_id)

    return y_pred, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    # Read filepaths
    filepaths = read_img_paths(args.path)

    # Get features
    features_base = dict()
    features2match = dict()
    
    for i in range(101, 111):
    
        features_base[str(i)]= {"sift":list(),
                                "orb" : list()}
        features2match[str(i)] = {"sift":list(),
                                  "orb" : list()}
        
        
        
    for path in tqdm(filepaths):
        gray = img_preprocessing(path)
        kp_sift, dsc_sift = get_features(gray, "sift")
        kp_orb, dsc_orb = get_features(gray, "orb")
        
        file_struct = path.split("/")
        file_name = file_struct[-1].split("_")
    
        if file_name[1].startswith('1'):
            features_base[file_name[0]]["sift"].append(dsc_sift)
            features_base[file_name[0]]["orb"].append(dsc_orb)
            
        else:
            features2match[file_name[0]]["sift"].append(dsc_sift)
            features2match[file_name[0]]["orb"].append(dsc_orb)


    # Match features
    preds, gt = match_fingerprints(features2match, features_base, "sift")
    print("--- SIFT ---")
    print(classification_report(gt, preds))

    preds, gt = match_fingerprints(features2match, features_base, "orb")
    print("--- ORB ---")
    print(classification_report(gt, preds))
