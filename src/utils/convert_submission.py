import json
import os

def shorten_image_id(image_id: str) -> str:
    """
    Shorten the image_id by keeping only the part before '_jpg'.
    Example: 'img201_jpg.rf.50c12877' -> 'img201'
    """
    if "_jpg" in image_id:
        return image_id.split("_jpg")[0]
    return image_id  


def convert_submission(input_path: str, output_path: str):
    """
    Convert submission_detection.json to submission_1.json
    with shortened image_id.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        for record in data:
            record["image_id"] = shorten_image_id(record["image_id"])
    elif isinstance(data, dict):
        data["image_id"] = shorten_image_id(data["image_id"])
    else:
        raise ValueError("Unsupported JSON structure in input file")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Converted submission saved at {output_path}")


if __name__ == "__main__":
    input_file = "outputs/submission_1.json"
    output_file = "outputs/submission_detection_1.json"

    convert_submission(input_file, output_file)
