import os
import pandas as pd
import numpy as np


def read_and_transform_data(data_filepath, dummy_filepath, output_dir):
    """
    读取数据文件和虚拟变量文件，对数据进行转换，并保存转换后的数据集。

    Parameters:
    - data_filepath: str, 数据文件的路径。
    - dummy_filepath: str, 虚拟变量文件的路径。
    - output_dir: str, 输出目录的路径。

    Returns:
    - transformed_data: np.ndarray, 转换后的数据集（如果需要的话）
    """
    try:
        df = pd.read_excel(data_filepath)
        with open(dummy_filepath, 'r') as file:
            dummy_flags = [int(i) for i in file.readline().strip().split(',')]

        nominal_attributes = []
        for i, is_nominal in enumerate(dummy_flags):
            if is_nominal:
                nominal_attributes.append(df.columns[i])
            else:
                df.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].min()) / (df.iloc[:, i].max() - df.iloc[:, i].min())

        dataset = pd.get_dummies(df, columns = nominal_attributes)

        # 假设'label'列是目标变量，且不需要进行独热编码
        if 'label' in dataset.columns:
            label = dataset['label']
            dataset = dataset.drop('label', axis = 1)
            dataset['label'] = label

        output_filepath = os.path.join(output_dir, os.path.splitext(os.path.basename(data_filepath))[0] + '.xlsx')
        dataset.to_excel(output_filepath, index=False)

        return dataset.values  # 返回numpy数组形式的数据集

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_filenames(directory, extension='.txt'):
    """
    获取指定目录下所有具有特定扩展名的文件名（不包括扩展名）。

    Parameters:
    - directory: str, 目录的路径。
    - extension: str, 文件扩展名（默认值为'.txt'）。

    Returns:
    - filenames: list, 文件名列表（不包括扩展名）。
    """
    return [f[:-len(extension)] for f in os.listdir(directory) if f.endswith(extension)]


if __name__ == '__main__':
    dataset_name = 'autoUniv-au7-500'
    dataset_dir = 'original_datasets'
    output_dir = 'transformed_datasets'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok = True)

    for name in get_filenames(os.path.join(dataset_dir, dataset_name), extension='.txt'):
        data_file = f"{name}.xlsx"
        dummy_file = f"{name}.txt"
        print(f"Processing {data_file} and {dummy_file}...")
        read_and_transform_data(
            os.path.join(dataset_dir, dataset_name, data_file),
            os.path.join(dataset_dir, dataset_name, dummy_file),
            output_dir
        )