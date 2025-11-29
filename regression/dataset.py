import os
import json
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple, Optional
from batchgenerators.utilities.file_and_folder_operations import isfile
from paths import regression_preprocessed
from torch.utils.data import DataLoader


class RegressionDataset:
    """
    支持五折交叉验证的NIfTI回归数据集，兼容推理（无split/无标签）模式。
    训练/验证：需split.json和txt标签。
    推理：无需split.json和标签，只遍历imagesTr下所有nii.gz。
    如果labelsTr存在且有对应标签，则返回label，否则不返回。
    """

    def __init__(self,
                 data_dir: str,
                 fold: int = 0,
                 mode: str = 'train',  # 'train', 'val', 'infer'
                 preprocess_fn: Optional[callable] = None,
                 transpose_fn: Optional[callable] = lambda x: np.transpose(x, (2, 1, 0))):
        """
        Args:
            data_dir: 数据集根目录，含imagesTr和labelsTr子文件夹
            fold: 当前使用的折（0-4）
            mode: 'train'/'val'/'infer'，指定当前模式
            preprocess_fn: 图像预处理函数（输入array，返回处理后array）
            transpose_fn: 图像转置函数（输入array，返回转置后array）
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "imagesTr")
        self.labels_dir = os.path.join(data_dir, "labelsTr")
        self.split_json_path = os.path.join(data_dir, "splits_final.json")
        self.fold = fold
        self.mode = mode
        self.preprocess_fn = preprocess_fn
        self.transpose_fn = transpose_fn

        # 验证参数合法性
        assert os.path.isdir(self.images_dir), f"图像目录不存在: {self.images_dir}"
        self.has_labels = os.path.isdir(self.labels_dir)
        if self.mode in ['train', 'val']:
            assert os.path.isfile(self.split_json_path), "训练/验证模式需split.json"
            self.identifiers = self._load_identifiers_from_split()
        else:
            # 推理模式，自动遍历imagesTr下所有nii.gz
            self.identifiers = [f[:-7] for f in os.listdir(self.images_dir) if f.endswith('.nii.gz')]
        self.identifiers.sort()

    def _load_identifiers_from_split(self) -> List[str]:
        """从split.json读取当前折的训练/验证样本标识符"""

        splits = self.load_json(self.split_json_path)
        assert len(splits) == 5, "split.json必须为5折划分"
        # 获取当前折的训练/验证标识符
        current_split = splits[self.fold]
        assert "train" in current_split and "val" in current_split, \
            f"split_{self.fold}缺少train/val字段"

        return current_split["train"] if self.mode == 'train' else current_split["val"]

    def _load_image(self, identifier: str) -> Tuple[np.ndarray, Dict]:
        """加载.nii.gz图像，返回数组和元数据（仿射矩阵、像素间距）"""
        image_path = os.path.join(self.images_dir, f"{identifier}.nii.gz")
        assert isfile(image_path), f"图像文件不存在: {image_path}"

        nifti_img = nib.load(image_path)
        image_array = nifti_img.get_fdata(dtype=np.float32)  # 转为float32

        if self.transpose_fn:
            image_array = self.transpose_fn(image_array)
        if self.preprocess_fn:
            image_array = self.preprocess_fn(image_array)

        print(image_array.shape)
        affine = nifti_img.affine
        spacing = np.diag(affine)[:3]  # 提取x,y,z轴像素间距

        return image_array, {"affine": affine, "spacing": spacing, "shape": image_array.shape}

    def _load_label(self, identifier: str) -> Optional[float]:
        """从txt文件加载回归标签（单个数字）"""
        label_path = os.path.join(self.labels_dir, f"{identifier}.txt")
        if isfile(label_path):
            return self.load_txt(label_path)
        return None

    def __getitem__(self, index: int) -> Dict:
        """按索引获取样本（兼容DataLoader）"""
        identifier = self.identifiers[index]
        image, _ = self._load_image(identifier)
        result = {
            "identifier": identifier,
            "image": image
        }
        if self.mode in ['train', 'val'] or self.has_labels:
            label = self._load_label(identifier)
            if label is not None:
                result["label"] = np.array([label], dtype=np.float32)
        return result

    def __len__(self) -> int:
        """返回样本数量"""
        return len(self.identifiers)

    def get_case(self, identifier: str) -> Dict:
        """按标识符获取样本"""
        if identifier not in self.identifiers:
            raise ValueError(f"标识符不在当前数据集: {identifier}")
        return self.__getitem__(self.identifiers.index(identifier))

    @staticmethod
    def load_txt(txt_path):
        with open(txt_path, 'r') as f:
            return float(f.read().strip())

    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)


class RegressionNpzDataset:
    """
    适用于预处理后npz文件的回归数据集，支持五折交叉验证和推理模式。
    训练/验证：需split.json和txt标签。
    推理：无需split.json和标签，只遍历目标shape文件夹下所有npz。
    如果labelsTr存在且有对应标签，则返回label，否则不返回。
    """

    def __init__(self,
                 dataset_id: int,
                 target_shape: tuple = (16, 648, 256),
                 fold: int = 0,
                 mode: str = 'train',  # 'train', 'val', 'infer'
                 npz_key: str = 'arr',
                 preprocess_fn: Optional[callable] = None):
        """
        Args:
            dataset_id: 数据集编号，如140
            target_shape: 目标shape, 如(16,648,256)
            fold: 当前使用的折（0-4）
            mode: 'train'/'val'/'infer'，指定当前模式
            npz_key: npz文件中保存的key
            preprocess_fn: 图像预处理函数（输入array，返回处理后array）
        """
        self.target_shape = target_shape
        self.target_shape_str = '_'.join(str(x) for x in target_shape)
        self.dataset_id = dataset_id
        self.dataset_name = f"Dataset{dataset_id}_WB"
        self.data_dir = os.path.join(regression_preprocessed, self.dataset_name)
        self.fingerprint_path = os.path.join(self.data_dir, self.target_shape_str, 'fingerprint.json')
        self.images_dir = os.path.join(self.data_dir, self.target_shape_str)
        self.labels_dir = os.path.join(self.data_dir, "labelsTr")
        self.split_json_path = os.path.join(self.data_dir, "splits_final.json")
        self.fold = fold
        self.mode = mode
        self.npz_key = npz_key
        self.preprocess_fn = preprocess_fn

        assert os.path.isdir(self.images_dir), f"图像目录不存在: {self.images_dir}"
        self.has_labels = os.path.isdir(self.labels_dir)
        if self.mode in ['train', 'val']:
            assert os.path.isfile(self.split_json_path), "训练/验证模式需split.json"
            self.identifiers = self._load_identifiers_from_split()
        else:
            # 推理模式，自动遍历目标shape文件夹下所有npz
            self.identifiers = [f[:-4] for f in os.listdir(self.images_dir) if f.endswith('.npz')]
        self.identifiers.sort()

    def _load_identifiers_from_split(self) -> List[str]:
        splits = self.load_json(self.split_json_path)
        assert len(splits) == 5, "split.json必须为5折划分"
        current_split = splits[self.fold]
        assert "train" in current_split and "val" in current_split, f"split_{self.fold}缺少train/val字段"
        return current_split["train"] if self.mode == 'train' else current_split["val"]

    def _load_npz(self, identifier: str) -> Tuple[np.ndarray, dict]:
        npz_path = os.path.join(self.images_dir, f"{identifier}.npz")
        assert os.path.isfile(npz_path), f"npz文件不存在: {npz_path}"
        arr = np.load(npz_path)[self.npz_key]
        if self.preprocess_fn:
            fingerprint = self.load_json(self.fingerprint_path)
            arr = self.preprocess_fn(arr, fingerprint['mean'], fingerprint['std'])
        return arr, {"shape": arr.shape}

    def _load_label(self, identifier: str) -> Optional[float]:
        label_path = os.path.join(self.labels_dir, f"{identifier}.txt")
        if os.path.isfile(label_path):
            return self.load_txt(label_path)/108.0
        return None

    def __getitem__(self, index: int) -> dict:
        identifier = self.identifiers[index]
        image, _ = self._load_npz(identifier)
        result = {"identifier": identifier, "image": image}
        if self.mode in ['train', 'val'] or self.has_labels:
            label = self._load_label(identifier)
            if label is not None:
                result["label"] = np.array([label], dtype=np.float32)
        return result

    def __len__(self) -> int:
        return len(self.identifiers)

    def get_case(self, identifier: str) -> dict:
        if identifier not in self.identifiers:
            raise ValueError(f"标识符不在当前数据集: {identifier}")
        return self.__getitem__(self.identifiers.index(identifier))

    @staticmethod
    def load_txt(txt_path):
        with open(txt_path, 'r') as f:
            return float(f.read().strip())

    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)




if __name__ == "__main__":
    # 测试RegressionNpzDataset
    npz_dataset = RegressionNpzDataset(
        dataset_id=140,
        target_shape=(16, 648, 256),
        fold=0,
        mode='train'
    )
    dataloader = DataLoader(npz_dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch["image"].shape)
        break
    # print(f"NPZ样本数量: {len(npz_dataset)}")
    # npz_sample = npz_dataset[0]
    # print(f"NPZ样本标识符: {npz_sample['identifier']}")
    # print(f"NPZ图像形状: {npz_sample['image'].shape}")
    # if 'label' in npz_sample:
    #     print(f"NPZ标签: {npz_sample['label']}")
