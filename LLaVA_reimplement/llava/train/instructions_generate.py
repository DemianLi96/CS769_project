import argparse
import json
import os
import openai
import time
import csv

NUM_SECONDS_TO_SLEEP = 0.5

def get_answer(content: str, max_tokens: int, temperature=0.2):
    """
    Calls OpenAI's Chat Completion API to generate a descriptive answer.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it using `export OPENAI_API_KEY=<your_api_key>` in your terminal."
        )

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for generating descriptive captions and instructions. Focus on details like size, shape, and ingredients."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Retrying...")
            time.sleep(NUM_SECONDS_TO_SLEEP)
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)

    raise RuntimeError("Failed to retrieve a response from the OpenAI API after multiple attempts.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate answers using GPT-4o.')
    parser.add_argument('--image-folder', type=str, default='./data', help='Folder containing input images')
    parser.add_argument('--labels-file', type=str, default='labels.txt', help='File containing image labels')
    parser.add_argument('--detections-file', type=str, default='detections.csv', help='CSV containing detections')
    parser.add_argument('--questions-file', type=str, default='question.jsonl', help='File containing questions')
    parser.add_argument('--output', type=str, default='answers_output.jsonl', help='Output file')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum number of tokens produced in the output')
    args = parser.parse_args()

    # Load labels
    image_to_label = {}
    with open(args.labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                img, label = parts
                image_to_label[img] = label

    # Load detections
    image_to_detections = {}
    with open(args.detections_file, 'r') as f:
        reader = csv.reader(f)
        # Assuming detections.csv has columns: image,category,bbox
        for row in reader:
            if len(row) < 3:
                continue
            img, category, bbox = row[0], row[1], row[2]
            if img not in image_to_detections:
                image_to_detections[img] = []
            image_to_detections[img].append(f'{category}: {bbox}')

    # Prepare output file
    if os.path.isfile(args.output):
        processed = [json.loads(line) for line in open(args.output)]
        processed_ids = set(item['id'] for item in processed)
    else:
        processed = []
        processed_ids = set()

    out_file = open(args.output, 'a')

    # Process questions
    with open(args.questions_file, 'r') as f:
        idx = len(processed) + 1
        for line in f:
            ques = json.loads(line)
            q_id = ques.get('id', idx)
            if q_id in processed_ids:
                continue

            image_name = ques.get('image')
            question_text = ques.get('question', '')

            labels_str = image_to_label.get(image_name, '')
            det_str = '\n'.join(image_to_detections.get(image_name, []))

            content = (
                f"[Context]\n"
                f"Image: {image_name}\n"
                f"Labels: {labels_str}\n"
                f"Detections:\n{det_str}\n\n"
                f"[Question]\n{question_text}\n\n"
            )

            answer = get_answer(content, args.max_tokens)
            result = {
                "id": q_id,
                "image": image_name,
                "question": question_text,
                "answer": answer
            }
            out_file.write(json.dumps(result) + "\n")
            out_file.flush()
            idx += 1

    out_file.close()
