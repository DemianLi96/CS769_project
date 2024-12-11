import json
import os


def get_images_with_json_content(json_path, image_folder):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    result = []
    for entry in data:
        image_name = entry['image']
        image_path = os.path.join(image_folder, image_name)

        if os.path.exists(image_path):
            result.append({
                'id': entry['id'],
                'image_path': image_path,
                'json_content_dict': entry
            })
        else:
            print(f"Image not found for id {entry['id']}: {image_name}")

    return result


if __name__ == "__main__":
    json_path = '../../data/test_reasoning_100.json'
    image_folder = '../../data/test_images'
    result = get_images_with_json_content(json_path, image_folder)

    for item in result:
        print(item)
