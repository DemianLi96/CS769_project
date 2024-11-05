### only keep the complex reasoning in the llava_instruct_80k.json

import json
import ijson


def load_json_as_dict(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        objects = ijson.items(file, 'item')
        for obj in objects:
            conversations_pairs = [
                (obj['conversations'][i], obj['conversations'][i + 1])
                for i in range(0, len(obj['conversations']) - 1, 2)
                if obj['conversations'][i]['from'] == 'human' and obj['conversations'][i + 1]['from'] == 'gpt'
            ]
            data_dict[obj['id']] = {
                'image': obj['image'],
                'conversations_pairs': conversations_pairs
            }
    return data_dict


def load_small_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def compare_conversations(large_data_dict, small_json, output_file_path):
    total_small_entry_count = 0
    total_pair_count = 0
    no_match_entry_count = 0
    not_found_pair_count = 0
    found_pair_count = 0
    found_entry_count = 0

    found_entries = []

    for small_entry in small_json:
        total_small_entry_count += 1
        if total_small_entry_count % 1000 == 0:
            print(f"Processed entries: {total_small_entry_count}")

        small_id = small_entry['id']

        small_conversations_pairs = [
            (small_entry['conversations'][i], small_entry['conversations'][i + 1])
            for i in range(0, len(small_entry['conversations']) - 1, 2)
            if small_entry['conversations'][i]['from'] == 'human' and small_entry['conversations'][i + 1]['from'] == 'gpt'
        ]

        large_conversations_data = large_data_dict.get(small_id)

        if large_conversations_data is None:
            no_match_entry_count += 1
            continue

        large_conversations_pairs = large_conversations_data['conversations_pairs']
        is_found = False
        found_conversations = []

        for small_pair in small_conversations_pairs:
            total_pair_count += 1
            if small_pair in large_conversations_pairs:
                is_found = True
                found_pair_count += 1
                found_conversations.extend(small_pair)

        if is_found:
            found_entry_count += 1
            found_entries.append({
                "id": small_id,
                "image": large_conversations_data['image'],
                "conversations": found_conversations
            })

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(found_entries, output_file, ensure_ascii=False, indent=2)

    print(f"Total small entries: {total_small_entry_count}")
    print(f"Not matched entries: {no_match_entry_count}")
    print(f"Total human-gpt pairs checked: {total_pair_count}")
    print(f"Not found pairs: {not_found_pair_count}")
    print(f"Found pairs: {found_pair_count}")
    print(f"Found entries: {found_entry_count}")

if __name__ == '__main__':
    large_json_path = '../../data/llava_instruct_80k.json'
    small_json_path = '../../data/complex_reasoning_77k.json'
    output_file_path = '../../data/found_entries.json'
    large_data_dict = load_json_as_dict(large_json_path)
    small_json = load_small_json(small_json_path)
    compare_conversations(large_data_dict, small_json, output_file_path)
