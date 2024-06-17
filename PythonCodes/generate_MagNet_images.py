import os.path
import random
from typing import Union
import numpy as np
import pandas as pd
import os
import cv2
import torch
import scipy.io as scio
from torchvision import transforms
from dataloader.magnet import MagNet
from dataloader.landmarks import detect_landmarks


AMP_LIST = [1.2, 1.4, 1.6, 1.8, 2.0,
            2.2, 2.4, 2.6, 2.8, 3.0]

parallel = 1
csv_path = r"B:\0_0NewLife\CASME_2\4classes.csv"
device = torch.device("cuda:0")
magnet = MagNet().to(device)
magnet.load_state_dict(torch.load(r"B:\0_0NewLife\0_Papers\FGRMER\weight\magnet.pt",
                                  map_location=device))
transforms = transforms.ToTensor()


def center_crop(img: np.array, crop_size: Union[tuple, int]) -> np.array:
    """Returns center cropped image

    Parameters
    ----------
    img : [type]
        Image to do center crop
    crop_size : Union[tuple, int]
        Crop size of the image

    Returns
    -------
    np.array
        Image after being center crop
    """
    width, height = img.shape[1], img.shape[0]

    # Height and width of the image
    mid_x, mid_y = int(width / 2), int(height / 2)

    if isinstance(crop_size, tuple):
        crop_width, crop_hight = int(crop_size[0] / 2), int(crop_size[1] / 2)
    else:
        crop_width, crop_hight = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[mid_y - crop_hight:mid_y + crop_hight, mid_x - crop_width:mid_x + crop_width]

    return crop_img


def unit_preprocessing(unit):
    unit = cv2.resize(unit, (256, 256))
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    unit = np.transpose(unit / 127.5 - 1.0, (2, 0, 1))
    unit = torch.FloatTensor(unit).unsqueeze(0)
    return unit


def unit_preprocessing_EVM(unit):
    unit = cv2.resize(unit, (256, 256))
    # unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    return unit


def magnify_postprocessing(unit):
    # Unnormalized the magnify images
    unit = unit[0].permute(1, 2, 0).contiguous()
    unit = (unit + 1.0) * 127.5

    # Convert back to images resize to (128, 128)
    unit = unit.numpy().astype(np.uint8)
    unit = cv2.cvtColor(unit, cv2.COLOR_RGB2GRAY)
    unit = cv2.resize(unit, (128, 128))
    return unit


def magnify_postprocessing_EVM(unit):
    # Convert back to images resize to (128, 128)
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2GRAY)
    unit = cv2.resize(unit, (128, 128))
    return unit


def unit_postprocessing(unit):
    unit = unit[0]

    # Normalized the images for each channels
    max_v = torch.amax(unit, dim=(1, 2), keepdim=True)
    min_v = torch.amin(unit, dim=(1, 2), keepdim=True)
    unit = (unit - min_v) / (max_v - min_v)

    # Sum up all the channels and take the average
    unit = torch.mean(unit, dim=0).numpy()

    # Resize to (128, 128)
    unit = cv2.resize(unit, (128, 128))
    return unit


def unit_postprocessing_EVM(unit):
    # 将三个通道的像素值取均值
    unit = np.mean(unit, axis=2)
    # Resize to (128, 128)
    unit = cv2.resize(unit, (128, 128))
    return unit


def get_patches(point: tuple):
    start_x = point[0] - 3
    end_x = point[0] + 4

    start_y = point[1] - 3
    end_y = point[1] + 4

    return start_x, end_x, start_y, end_y


def create_patches(image_root, catego, out_dir):
    # Label for the image
    data_info = pd.read_csv(csv_path,
                            dtype={
                                "Subject": str,
                                "Filename": str
                            })
    # for index in data_info.shape[0]:
    for amp_factor in AMP_LIST:
        for index in range(data_info.shape[0]):

            subject = data_info.loc[index, "Subject"]
            onset_name = data_info.loc[index, "OnsetFrame"]
            apex_name = data_info.loc[index, "ApexFrame"]
            folder = data_info.loc[index, "Filename"]

            # Create the path for onset frame and apex frame
            if catego == "SAMM":
                onset_path = f"{image_root}/{subject}/{folder}/{subject}_{onset_name:05}.jpg"
                apex_path = f"{image_root}/{subject}/{folder}/{subject}_{apex_name:05}.jpg"
            else:
                onset_path = f"{image_root}/sub{subject.zfill(2)}/{folder}/img{onset_name}.jpg"
                apex_path = f"{image_root}/sub{subject.zfill(2)}/{folder}/img{apex_name}.jpg"

            # Read in the image
            onset_frame = cv2.imread(onset_path)
            assert onset_frame is not None, f"{onset_path} not exists"
            apex_frame = cv2.imread(apex_path)
            assert apex_frame is not None, f"{apex_path} not exists"

            if catego == "SAMM":
                onset_frame = center_crop(onset_frame, (420, 420))
                apex_frame = center_crop(apex_frame, (420, 420))

            # Preprocessing of the image
            onset_frame = unit_preprocessing(onset_frame).to(device)
            apex_frame = unit_preprocessing(apex_frame).to(device)

            with torch.no_grad():
                print(subject, folder, amp_factor)
                # 这里有一点没考虑好：这里把amp_factor在每一个 图像中都随机取了一个值
                # 因此会产生N路图像放大倍数不相同的情况，应该把amp_factor放在for循环外面
                # 但是感觉好像影响不大，待会试试看吧

                # Get the magnify results
                shape_representation, magnify = magnet(batch_A=onset_frame,
                                                       batch_B=apex_frame,  # apex_frame
                                                       batch_C=None,
                                                       batch_M=None,
                                                       amp_factor=amp_factor,
                                                       mode="evaluate")

                # Do the post processing the transform back to numpy
                magnify = magnify_postprocessing(magnify.to("cpu"))
                shape_representation = unit_postprocessing(shape_representation.to("cpu"))

                # Landmarks detection
                points = detect_landmarks(magnify)

                patches = []
                for point in points:
                    start_x, end_x, start_y, end_y = get_patches(point)
                    patches.append(
                        transforms(np.expand_dims(shape_representation[start_x:end_x, start_y:end_y], axis=-1))
                    )
                patches = torch.cat(patches, dim=0)  # [30, 7, 7]
            print(f"Amp_factor {amp_factor} for {subject}_{folder} is accomplished.")
            file_dir = f"{out_dir}/Inter_{parallel}/"
            os.makedirs(file_dir, exist_ok=True)
            file_name = f"{file_dir}/sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
            torch.save(patches, file_name)


if __name__ == '__main__':
    out_dir = r'B:\0_0NewLife\0_Papers\FGRMER\CASME2\mat\MagNet'
    # image_root = f"B:/0_0NewLife/0_Papers/FGRMER/CASME2/Interpolation/Inter_{parallel+1}"
    image_root = f"B:\\0_0NewLife\\CASME_2\\RAW_selected"
    create_patches(image_root, "CASME", out_dir)
