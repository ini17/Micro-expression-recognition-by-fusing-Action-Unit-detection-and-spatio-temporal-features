import numpy as np
import pandas as pd
import cv2
import os
import torch


image_root = r"B:\0_0NewLife\0_Papers\SMC\CASME2\Interpolation\Cropped_AU"
out_dir = r"B:\0_0NewLife\0_Papers\SMC\CASME2\AUfeature"
csv_path = r"B:\0_0NewLife\datasets\CASME_2\4classes.csv"
catego = "CASME"

AU_number = 19  # CASME 2
size = 144

AU_CODE = [1, 2, 4, 5,
           6, 7, 9,
           10, 12, 14, 15,
           16, 17, 18, 20,
           24, 25, 26, 38]
AU_DICT = {
    number: idx
    for idx, number in enumerate(AU_CODE)
}


def raw_preprocessing(unit):
    unit = cv2.resize(unit, (size, size))
    unit = np.transpose(unit, (2, 0, 1))
    unit = torch.FloatTensor(unit)
    return unit


def image_crop(csv_frame):
    for idx in range(csv_frame.shape[0]):
        # 生成可供P3D训练的数据
        print("="*19 + f"Line {idx+1} processing" + "="*19)
        subject = csv_frame.loc[idx, "Subject"]
        folder = csv_frame.loc[idx, "Filename"]

        frames = torch.empty((3, 16, size, size))
        for i in range(16):
            frames[:, i] = raw_preprocessing(cv2.imread(
                f"{image_root}/sub{subject.zfill(2)}/{folder}/img{i}.jpg"
            ))
        torch.save(frames, f"{out_dir}/sub{subject.zfill(2)}_{folder}.pt")


def main():
    os.makedirs(out_dir, exist_ok=True)
    csv_file = pd.read_csv(csv_path, dtype={"Subject": str})
    image_crop(csv_file)


if __name__ == "__main__":
    main()
