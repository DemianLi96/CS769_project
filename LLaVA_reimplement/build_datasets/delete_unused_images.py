## clean unused COCO images

import json
import os

def get_images_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    image_names = {entry['image'] for entry in data}
    return image_names

def delete_unused_images(folder_path, valid_images):
    for filename in os.listdir(folder_path):
        if filename not in valid_images:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted: {filename}")



if __name__ == "__main__":
    json_path = '../../data/complex_reasoning_20k.json'
    folder_path = '../../data/train2014'
    valid_images = get_images_from_json(json_path)
    delete_unused_images(folder_path, valid_images)
