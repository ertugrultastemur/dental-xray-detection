# -*- coding: utf-8 -*-
# @date: 6.05.2024
# @author: ikbal
# @file: conv.py


import pandas as pd
import os


def convert_annotations_to_yolo(data_path, output_folder):
    # read csv file
    df = pd.read_csv(data_path)

    # get unique classes and create class index mapping
    classes = sorted(df['class'].unique())
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # make output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # processing each row
    for _, row in df.iterrows():
        filename = row['filename']
        width, height = row['width'], row['height']
        cls = class_to_index[row['class']]
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        # normalize bounding box coordinates
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        # create txt file
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)

        with open(txt_path, 'a') as file:
            file.write(f"{cls} {x_center} {y_center} {w} {h}\n")

    return class_to_index


def main():
    dataset_root = 'datad'

    # define sets
    sets = ['train', 'valid', 'test']
    class_list = None

    for set_name in sets:
        csv_path = os.path.join(dataset_root, f'{set_name}/_annotations.csv')
        output_path = os.path.join(dataset_root, f'{set_name}/labels')

        # transform annotations to yolo format
        if class_list is None:
            class_list = convert_annotations_to_yolo(csv_path, output_path)
        else:
            convert_annotations_to_yolo(csv_path, output_path)

        print(f"{set_name}")

    print("Class Index Mapping:")
    print(class_list)


if __name__ == "__main__":
    main()
