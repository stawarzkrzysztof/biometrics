import cv2
import os
import numpy as np
from typing import List
from random import randint
from tqdm import tqdm


def floodfill_mask(img: np.ndarray, kernel_size: int, threshold=1) -> np.ndarray:

    img_floodfill: np.ndarray = img.copy()

    h, w = img_floodfill.shape[:2]

    fl_mask: np.array = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(image=img_floodfill, mask=fl_mask, seedPoint=(10, 10), newVal=255)

    dil_kernel: np.array = np.ones((kernel_size, kernel_size), np.uint8)
    mask: np.ndarray = cv2.bitwise_not(cv2.dilate(img_floodfill, dil_kernel))

    mask[mask > threshold] = 255

    return mask


def enhance_contrast(img: np.ndarray) -> np.ndarray:

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img_inversed_enhanced = 255 - clahe.apply(img)

    img_inversed_enhanced_twice: np.ndarray = clahe.apply(img_inversed_enhanced)
    
    return img_inversed_enhanced_twice


def find_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:

    img_combined = cv2.bitwise_and(img, mask)

    img_med_blured = cv2.bitwise_not(cv2.medianBlur(img_combined, ksize=5))

    img_morph = cv2.bitwise_not(cv2.morphologyEx(img_med_blured, op=cv2.MORPH_CLOSE, kernel=(13, 13)))

    return img_morph


def find_vessel_skeleton(img: np.ndarray) -> np.ndarray:
    skel_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel: np.ndarray = np.zeros(img.shape, np.uint8)

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, skel_element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, skel_element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


def main(path):
    file_names: List[str] = [os.path.join(path, image) for image in os.listdir(path)]
    file_names = sorted(file_names, key=lambda n: (int(n.split("_")[0][-1]), int(n.split("_")[1][:-4])))

    for file in tqdm(file_names):

        img = cv2.imread(file)

        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = floodfill_mask(img_grey, kernel_size=13)
        
        img_clahe_inversed = enhance_contrast(img_grey)

        gauss_kernel = 7
        gauss_sigma = 0.3*((gauss_kernel-1)*0.5-1)+0.8

        img_blured: np.ndarray = cv2.GaussianBlur(img_clahe_inversed,
                                                  ksize=(gauss_kernel, gauss_kernel),
                                                  sigmaX=gauss_sigma, sigmaY=gauss_sigma)

        img_adptv_thr: np.ndarray = cv2.adaptiveThreshold(img_blured, maxValue=255,
                                                          adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          thresholdType=cv2.THRESH_BINARY, blockSize=5, C=0)

        output_img = find_features(img_adptv_thr, mask)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(output_img, connectivity=8)
        # sizes = stats[1:, -1]
        nb_components -= 1

        img_vessel_skel = find_vessel_skeleton(output_img)

        # img_skel_name: str = "skeleton_" + file.split('/')[-1]
        # cv2.imwrite(os.path.join("/Users/stawager/Studia/s4/biometrics/4/skeletons", img_skel_name), img_vessel_skel)

        # img_clahe_inversed = cv2.cvtColor(img_clahe_inversed, cv2.COLOR_GRAY2RGB)
        # img_vessel_skel = cv2.cvtColor(img_vessel_skel, cv2.COLOR_GRAY2RGB)

        img_vessel_skel2 = img_vessel_skel.copy()

        for i in range(1, nb_components + 1):
            x, y, w, h, area = stats[i]
            if 500 < area < 5_000:
                # print(f"{file}", x, y, w, h)
                cv2.rectangle(img_vessel_skel2, (x, y), (x + w, y + h),
                              (randint(50, 255), randint(50, 255), randint(50, 255)), 2)

        # concat_images1 = cv2.hconcat([img, img_clahe_inversed])
        # concat_images2 = cv2.hconcat([img_vessel_skel, img_vessel_skel2])
        # full_image = cv2.vconcat([concat_images1, concat_images2])
        cv2.imshow(f"{file.split('/')[-1]}", img_vessel_skel)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
        # full_image_name: str = "full_image_" + file.split('/')[-1]
        # cv2.imwrite(os.path.join("/Users/stawager/Studia/s4/biometrics/4/full_images", full_image_name), full_image)


if __name__ == "__main__":
    data_path = "/Users/stawager/Studia/s4/biometrics/4/RIDB"
    main(data_path)
    quit()
