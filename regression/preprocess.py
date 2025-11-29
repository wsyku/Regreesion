import os
import numpy as np
import nibabel as nib
import shutil
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice
import json
from paths import regression_raw,regression_preprocessed


def crop_to_nonzero(data):
    """
    对数据进行裁剪,只保留非零区域。
    :param data: 输入数据,形状为(1, d, h, w)或(d, h, w)
    :return: 裁剪后的数据,形状为(d', h', w')和边界框bbox
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    if data.ndim == 4:
        data = data.squeeze(0)  # 去掉通道维度,变为(d, h, w)

    # 计算所有为True区域的最小外接边界框（bounding box）。返回每个维度的起止索引，格式为[(min_x, max_x), (min_y, max_y), (min_z, max_z)]
    bbox = get_bbox_from_mask(data)
    # 将边界框（bbox）的起止索引转换为Python切片（slice）对象，返回tuple([slice(min_x, max_x), slice(min_y, max_y), slice(min_z, max_z)])
    slicer = bounding_box_to_slice(bbox)
    data = data[slicer]  # 非零掩码(裁剪后,维度与data相同)
    return data, bbox  # 返回裁剪后的数据、边界框


def pad_to_shape(array, target_shape, mode='constant', constant_values=0, crop=True):
    """
    将三维数组array对称padding到target_shape。
    :param array: 输入三维ndarray，形状为(d, h, w)
    :param target_shape: 目标形状，tuple/list，(d_t, h_t, w_t)
    :param mode: 填充方式，默认为'constant'，可选'edge'、'reflect'等，参考np.pad文档
    :param constant_values: 填充常数值，仅在mode='constant'时有效
    :param crop: 是否先裁剪非零区域（默认True）
    :return: padding后的三维ndarray，形状为target_shape
    """
    if crop:
        array, _ = crop_to_nonzero(array)
        print(array.shape)
    assert array.ndim == 3, "输入必须是三维数组"
    assert len(target_shape) == 3, "目标形状必须是三元组"
    pads = []
    for i in range(3):
        total_pad = target_shape[i] - array.shape[i]
        print(total_pad)

        assert total_pad >= 0, f"目标形状的第{i}维必须大于等于输入形状"
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pads.append((pad_before, pad_after))
    if mode == 'constant':
        padded = np.pad(array, pads, mode=mode, constant_values=constant_values)
    else:
        padded = np.pad(array, pads, mode=mode)
    return padded


def calculate_npz_mean_std(npz_folder, npz_key='arr', json_save_dir=None, verbose=False):
    """
    计算npz_folder下所有npz文件的均值和标准差，并保存为json文件。
    :param npz_folder: npz文件夹路径
    :param npz_key: npz文件中数组的key
    :param json_save_dir: json保存目录，默认为npz_folder
    :param verbose: 是否打印日志
    :return: 均值和标准差
    """
    values = []
    for fname in os.listdir(npz_folder):
        if fname.endswith('.npz'):
            arr = np.load(os.path.join(npz_folder, fname))[npz_key]
            values.append(arr.flatten())
    if not values:
        raise ValueError(f"No npz files found in {npz_folder}")
    all_values = np.concatenate(values)
    mean = float(np.mean(all_values))
    std = float(np.std(all_values))
    # 保存json
    if json_save_dir is None:
        json_save_dir = npz_folder
    os.makedirs(json_save_dir, exist_ok=True)
    json_path = os.path.join(json_save_dir, 'fingerprint.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'mean': mean, 'std': std}, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"Mean: {mean}, Std: {std}, saved to {json_path}")
    return mean, std


def process_dataset(dataset_id, target_shape=(16, 648, 256), npz_key='arr', verbose=False):
    """
    对regression_raw/Dataset{dataset_id}_WB/下的imagesTr文件夹内所有nii.gz文件应用process_func，保存为npz文件，
    并将与imagesTr同目录的其他文件一并移动到目标目录。
    regression_raw会被替换为regression_preprocessed。
    :param dataset_id: 数据集编号
    :param target_shape: 目标shape
    :param npz_key: npz文件中保存的key
    :param verbose: 是否打印日志
    """
    dataset_name = f"Dataset{dataset_id}_WB"
    root_folder = os.path.join(regression_raw, dataset_name)
    imagesTr_dir = os.path.join(root_folder, 'imagesTr')
    assert os.path.isdir(imagesTr_dir), f"{imagesTr_dir} 不存在"
    target_shape_str = '_'.join(str(x) for x in target_shape)
    target_root = os.path.join(regression_preprocessed, dataset_name)
    target_imagesTr_dir = os.path.join(target_root, target_shape_str)
    if os.path.exists(target_imagesTr_dir):
        if verbose:
            print(f"Shape folder {target_imagesTr_dir} already exists, skipping processing.")
        return
    os.makedirs(target_imagesTr_dir, exist_ok=True)

    # 处理imagesTr下的nii.gz文件
    for fname in os.listdir(imagesTr_dir):
        if fname.endswith('.nii.gz'):
            src_path = os.path.join(imagesTr_dir, fname)
            arr = nib.load(src_path).get_fdata()
            arr = np.transpose(arr, (2, 1, 0))
            arr = pad_to_shape(arr, target_shape)
            npz_name = fname.replace('.nii.gz', '.npz')
            npz_path = os.path.join(target_imagesTr_dir, npz_name)
            np.savez_compressed(npz_path, **{npz_key: arr})
            if verbose:
                print(f"Saved: {npz_path}")

    # 复制imagesTr同目录下的其他文件/文件夹（除了imagesTr本身）
    for item in os.listdir(root_folder):
        src_item = os.path.join(root_folder, item)
        dst_item = os.path.join(target_root, item)
        if item != 'imagesTr':
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                if verbose:
                    print(f"Copied folder: {src_item} -> {dst_item}")
            else:
                shutil.copy2(src_item, dst_item)
                if verbose:
                    print(f"Copied file: {src_item} -> {dst_item}")
    # 计算均值和标准差并保存json
    calculate_npz_mean_std(target_imagesTr_dir, npz_key=npz_key, verbose=verbose)


def Zscore_normalize(arr: np.ndarray,mean:float,std:float) -> np.ndarray:
    """
    对数组进行Z-score归一化。
    """
    return (arr - mean) / std if std != 0 else arr



if __name__ == "__main__":
    # 测试：以140号数据集和默认shape为例
    process_dataset(140, target_shape=(16, 648, 256), npz_key='arr')
    print("处理完成！")
