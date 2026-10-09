"""TrustMR experiments: prepare COCO image-caption pairs for multimodal encoding."""

import json
from pathlib import Path
import pandas as pd

COCO_ROOT = Path("data/raw/coco")
OUTPUT_PARQUET = Path("data/raw/coco_pairs.parquet")

def load_coco_captions(annotation_file, image_dir, split_name):
    """Select the first caption for each image in annotation order."""
    print(f"\n[{split_name}] Loading {annotation_file}...")
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img['file_name'] for img in data['images']}
    print(f"  Total images: {len(images)}")
    
    captions_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in captions_by_image:
            captions_by_image[img_id] = []
        captions_by_image[img_id].append(ann['caption'])
    print(f"  Total captions: {sum(len(c) for c in captions_by_image.values())}")
    
    pairs = []
    for img_id, file_name in images.items():
        if img_id not in captions_by_image:
            continue
        caption = captions_by_image[img_id][0].strip()
        if len(caption) < 5:
            continue
        pairs.append({
            'image_id': img_id,
            'image_path': str(image_dir / file_name),
            'caption': caption,
            'split': split_name,
        })
    
    print(f"  Valid pairs: {len(pairs)}")
    return pairs


def main():
    train_pairs = load_coco_captions(
        annotation_file=COCO_ROOT / "annotations" / "captions_train2017.json",
        image_dir=COCO_ROOT / "train2017",
        split_name="train",
    )
    val_pairs = load_coco_captions(
        annotation_file=COCO_ROOT / "annotations" / "captions_val2017.json",
        image_dir=COCO_ROOT / "val2017",
        split_name="val",
    )
    
    all_pairs = train_pairs + val_pairs
    df = pd.DataFrame(all_pairs)
    
    print(f"\nTotal pairs: {len(df)}")
    sample = df.sample(min(100, len(df)), random_state=42)
    missing = [p for p in sample['image_path'] if not Path(p).exists()]
    if missing:
        print(f"WARNING: {len(missing)}/100 sampled images are missing!")
        print(f"  Example missing: {missing[0]}")
    else:
        print(f"All 100 sampled images verified to exist.")
    
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved to {OUTPUT_PARQUET}")
    print(f"  File size: {OUTPUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\nSample records:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
