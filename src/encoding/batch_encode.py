"""TrustMR experiments: ImageBind encoding of COCO images and captions."""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data


DEFAULT_INPUT_PARQUET = "data/raw/coco_pairs.parquet"
DEFAULT_OUTPUT_HDF5 = "data/embeddings/coco_imagebind.h5"
DEFAULT_IMAGE_BATCH_SIZE = 32
DEFAULT_TEXT_BATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PARQUET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_HDF5)
    parser.add_argument("--image_batch_size", type=int, default=DEFAULT_IMAGE_BATCH_SIZE)
    parser.add_argument("--text_batch_size", type=int, default=DEFAULT_TEXT_BATCH_SIZE)
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N samples (for testing). Default: all.")
    return parser.parse_args()


def load_model(device):
    print("Loading ImageBind huge model...")
    t0 = time.time()
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print(f"Model loaded in {time.time()-t0:.1f}s on {device}")
    return model


@torch.no_grad()
def encode_images(model, image_paths, batch_size, device):
    embeddings = []
    n = len(image_paths)
    failed_indices = []
    
    for i in tqdm(range(0, n, batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        try:
            batch_input = imagebind_data.load_and_transform_vision_data(batch_paths, device)
            inputs = {ModalityType.VISION: batch_input}
            outputs = model(inputs)
            batch_emb = outputs[ModalityType.VISION].cpu().numpy()
            embeddings.append(batch_emb)
        except Exception as e:
            print(f"\n[ERROR] batch {i}-{i+batch_size}: {type(e).__name__}: {e}")
            embeddings.append(np.zeros((len(batch_paths), 1024), dtype=np.float32))
            failed_indices.extend(range(i, i + len(batch_paths)))
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    embeddings = embeddings / norms
    
    if failed_indices:
        print(f"  WARNING: {len(failed_indices)} samples failed, using zero vectors")
    
    return embeddings.astype(np.float32), failed_indices


@torch.no_grad()
def encode_texts(model, texts, batch_size, device):
    embeddings = []
    failed_indices = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_input = imagebind_data.load_and_transform_text(batch_texts, device)
            inputs = {ModalityType.TEXT: batch_input}
            outputs = model(inputs)
            batch_emb = outputs[ModalityType.TEXT].cpu().numpy()
            embeddings.append(batch_emb)
        except Exception as e:
            print(f"\n[ERROR] batch {i}-{i+batch_size}: {type(e).__name__}: {e}")
            embeddings.append(np.zeros((len(batch_texts), 1024), dtype=np.float32))
            failed_indices.extend(range(i, i + len(batch_texts)))
    
    embeddings = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    embeddings = embeddings / norms
    
    if failed_indices:
        print(f"  WARNING: {len(failed_indices)} samples failed, using zero vectors")
    
    return embeddings.astype(np.float32), failed_indices


def sanity_check(image_emb, text_emb, n_samples=1000):
    """Compare matched vs random-pair cosine similarity as a quick embedding check."""
    n = len(image_emb)
    sample_idx = np.random.RandomState(42).choice(n, min(n_samples, n), replace=False)
    
    matched = (image_emb[sample_idx] * text_emb[sample_idx]).sum(axis=1)
    
    perm = np.random.RandomState(123).permutation(sample_idx)
    mismatched = (image_emb[sample_idx] * text_emb[perm]).sum(axis=1)
    
    print("\n" + "=" * 60)
    print("Sanity check (cosine similarity, expecting matched >> random):")
    print(f"  Matched pair mean: {matched.mean():.4f} (std: {matched.std():.4f})")
    print(f"  Random pair mean:  {mismatched.mean():.4f} (std: {mismatched.std():.4f})")
    
    if matched.mean() - mismatched.mean() < 0.1:
        print(f"  [WARNING] matched not significantly higher than random!")
    else:
        print(f"  [PASSED] matched pairs clearly more similar.")
    print("=" * 60)


def save_hdf5(output_path, image_emb, text_emb, df, failed_img, failed_txt):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    
    dt = h5py.string_dtype(encoding='utf-8')
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('image_embeddings', data=image_emb, compression='gzip')
        f.create_dataset('text_embeddings', data=text_emb, compression='gzip')
        f.create_dataset('image_ids', data=df['image_id'].values)
        f.create_dataset('image_paths',
                         data=np.array(df['image_path'].values, dtype=object), dtype=dt)
        f.create_dataset('captions',
                         data=np.array(df['caption'].values, dtype=object), dtype=dt)
        f.create_dataset('splits',
                         data=np.array(df['split'].values, dtype=object), dtype=dt)
        if failed_img:
            f.create_dataset('failed_image_indices', data=np.array(failed_img))
        if failed_txt:
            f.create_dataset('failed_text_indices', data=np.array(failed_txt))
        
        f.attrs['model'] = 'ImageBind-huge'
        f.attrs['embedding_dim'] = 1024
        f.attrs['n_samples'] = len(df)
        f.attrs['l2_normalized'] = True
    
    size_gb = output_path.stat().st_size / 1024**3
    print(f"Saved {len(df)} pairs ({size_gb:.2f} GB)")


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Configuration:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Image batch size: {args.image_batch_size}")
    print(f"  Text batch size:  {args.text_batch_size}")
    print(f"  Limit: {args.n if args.n else 'all'}")
    print(f"  Device: {device}\n")
    
    df = pd.read_parquet(args.input)
    if args.n:
        df = df.head(args.n).reset_index(drop=True)
    print(f"Loaded {len(df)} pairs to encode\n")

    model = load_model(device)

    t0 = time.time()
    image_emb, failed_img = encode_images(
        model, df['image_path'].tolist(),
        args.image_batch_size, device,
    )
    t_img = time.time() - t0
    print(f"Image encoding: {t_img:.1f}s ({len(df)/t_img:.1f} samples/sec)\n")

    t0 = time.time()
    text_emb, failed_txt = encode_texts(
        model, df['caption'].tolist(),
        args.text_batch_size, device,
    )
    t_txt = time.time() - t0
    print(f"Text encoding: {t_txt:.1f}s ({len(df)/t_txt:.1f} samples/sec)")

    sanity_check(image_emb, text_emb)
    save_hdf5(args.output, image_emb, text_emb, df, failed_img, failed_txt)


if __name__ == "__main__":
    main()
