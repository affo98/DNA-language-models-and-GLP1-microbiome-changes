import os
import json


def main():
    with open("unseen_bio_lookup.json") as file:
        id_map = json.load(file)
    print(id_map)
    base_dir = "UNSEEN_BIO"

    for old_id, new_id in id_map.items():
        old_folder_path = os.path.join(base_dir, old_id)
        new_folder_path = os.path.join(base_dir, new_id)

        if not os.path.isdir(old_folder_path):
            print(f"Warning: {old_folder_path} does not exist. Skipping.")
            continue

        for suffix in ["1", "2"]:
            old_filename = f"{old_id}_{suffix}.fastq.gz"
            new_filename = f"{new_id}_{suffix}.fastq.gz"
            old_file_path = os.path.join(old_folder_path, old_filename)
            new_file_path = os.path.join(old_folder_path, new_filename)

            if os.path.exists(old_file_path):
                os.rename(old_file_path, new_file_path)
            else:
                print(f"Warning: {old_file_path} not found. Skipping.")

        os.rename(old_folder_path, new_folder_path)

    print("Renaming complete.")


if __name__ == "__main__":
    main()
