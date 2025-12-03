#!/usr/bin/env python
import os
import cv2
import numpy as np
import torch
import matplotlib.image as mpimg
from PIL import Image
import gc

# --------------------
# Configuration
# --------------------
N_IMAGES = 202599
IMG_SIZE = 256
IMG_DIR = "data/img_align_celeba"
FINAL_IMG_PATH = f"images_{IMG_SIZE}_{IMG_SIZE}.pth"
ATTR_PATH = "attributes.pth"

# Taille des batchs (changer si besoin)
BATCH_SIZE = 5000


# ============================================================
#                 PREPROCESS IMAGES — BATCHED
# ============================================================
def preprocess_images_batch(batch_size=BATCH_SIZE):

    if os.path.isfile(FINAL_IMG_PATH):
        print(f"{FINAL_IMG_PATH} already exists, skipping image preprocessing.")
        return

    print("=== Batched Image Preprocessing ===")
    print(f"Batch size: {batch_size}")
    total = N_IMAGES
    n_batches = (total + batch_size - 1) // batch_size
    print(f"Total images: {N_IMAGES}")
    print(f"Total batches: {n_batches}")

    for b in range(n_batches):
        start = b * batch_size
        end = min((b+1) * batch_size, total)

        print(f"Batch {b+1}/{n_batches}")

        batch_images = []

        for i in range(start + 1, end + 1):
            img_path = os.path.join(IMG_DIR, f"{i:06d}.jpg")
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)[20:-20]
            batch_images.append(img)

        resized = []
        for img in batch_images:
            r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            resized.append(r)

        tensor = torch.from_numpy(
            np.stack([img.transpose(2,0,1) for img in resized], axis=0)
        )

        torch.save(tensor, f"images_{IMG_SIZE}_{IMG_SIZE}_part{b}.pth")

        # -------- CLEAN RAM --------
        del batch_images, resized, tensor
        gc.collect()
        torch.cuda.empty_cache()

    print("\nBatched preprocessing DONE.")
    print("Run merge_images() to create the final file.")


# ============================================================
#                     MERGE ALL BATCHES
# ============================================================
def merge_images():

    print("\n=== Merging all batch files ===")

    parts = []
    batch_id = 0

    while True:
        fname = f"images_{IMG_SIZE}_{IMG_SIZE}_part{batch_id}.pth"
        if not os.path.isfile(fname):
            break

        print(f"Loading {fname} ...")
        part = torch.load(fname)
        parts.append(part)
        batch_id += 1

    if batch_id == 0:
        print("No part files found. Did you run preprocess_images_batch()?")
        return

    print(f"Found {batch_id} batches. Concatenating...")
    full = torch.cat(parts, dim=0)
    print("Final tensor shape:", full.size())

    print(f"Saving merged file to {FINAL_IMG_PATH} ...")
    torch.save(full, FINAL_IMG_PATH)

    print("Merge completed successfully!")


# ============================================================
#                     PREPROCESS ATTRIBUTES
# ============================================================
def preprocess_attributes():

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    attr_lines = [line.rstrip() for line in open('list_attr_celeba.txt', 'r')]
    assert len(attr_lines) == N_IMAGES + 2

    attr_keys = attr_lines[1].split()
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

    for i, line in enumerate(attr_lines[2:]):
        image_id = i + 1
        split = line.split()
        assert len(split) == 41
        assert split[0] == ('%06i.jpg' % image_id)
        assert all(x in ['-1', '1'] for x in split[1:])
        for j, value in enumerate(split[1:]):
            attributes[attr_keys[j]][i] = value == '1'

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)


# ============================================================
#                          MAIN
# ============================================================
preprocess_images_batch()
merge_images()
preprocess_attributes()
