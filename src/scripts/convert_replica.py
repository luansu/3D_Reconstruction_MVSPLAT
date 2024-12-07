''' Build upon: https://github.com/dcharatan/real_estate_10k_tools
                https://github.com/donydchen/matchnerf/blob/main/datasets/dtu.py 
    DTU Acquired instruction: https://github.com/donydchen/matchnerf?tab=readme-ov-file#dtu-for-both-training-and-testing'''

import subprocess
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
import argparse
from tqdm import tqdm
import json

import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input dtu raw directory")
parser.add_argument("--output_dir", type=str, help="output directory")
args = parser.parse_args()

INPUT_IMAGE_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)


# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)

#Checked
def build_camera_info(id_list, root_dir):
    """Return the camera information for the given id_list"""
    intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
    scale_factor = 1.0 / 200
    downSample = 1.0

    proj_mat_filename = os.path.join(
            root_dir, f"traj_w_c.txt")
    intrinsic, extrinsic, near_far = read_cam_file(proj_mat_filename)
    intrinsic[:2] *= 4
    intrinsic[:2] = intrinsic[:2] * downSample
    for i, vid in enumerate(id_list):
        intrinsics[vid] = intrinsic
        extrinsic[i][:3, 3] *= scale_factor
        world2cams[vid] = extrinsic[i]
        cam2worlds[vid] = np.linalg.inv(extrinsic[i])
        near_fars[vid] = near_far

    return intrinsics, world2cams, cam2worlds, near_fars

# checked
def read_cam_file(filename):
    '''
        Read Extrinstic info
        return list
    '''
    scale_factor = 1.0 / 200

    with open(filename) as f:
        lines = [np.array(line.split(), dtype=np.float64) for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsic = np.array(lines).reshape(-1, 4, 4)
    # intrinsics: line [7-10), 3x3 matrix
    '''
        "w": 1200,
        "h": 680,
        "fx": 600.0,
        "fy": 600.0,
        "cx": 599.5,
        "cy": 339.5,
        depth_range: [0.0, 8.0]
            '''
    intrinsic = np.array([[600.0, 0.0, 599.5],
                          [0.0, 600.0, 339.5],
                          [0.0, 0.0, 1.0]])
    # depth_min & depth_interval: line 11
    depth_min = 0.0 * scale_factor
    depth_max = depth_min + 8.0 * 192 * scale_factor
    near_far = [depth_min, depth_max]
    return intrinsic, extrinsic, near_far

#Uncheckel
def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    """ Extracted from: https://github.com/donydchen/matchnerf/blob/main/configs/dtu_meta/val_all.txt """
    '''
        File này dùng để trích xuất ra các khóa, tương ứng với các cảnh/phòng trong dataset
    '''
    keys = [
        "room_0"
    ]
    print(f"Found {len(keys)} keys.")
    return keys

# unchecked
def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))

def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    '''
        Load raw image file
    '''
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))

# checked
def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    for cur_id in range(256):
        cur_image_name = f"rgb_{cur_id}.png"
        img_bin = load_raw(example_path / cur_image_name)
        images_dict[cur_id] = img_bin

    return images_dict


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, "camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]

# checked
def load_metadata(intrinsics, world2cams) -> Metadata:
    timestamps = []
    cameras = []
    url = ""

    for vid, intr in intrinsics.items():
        timestamps.append(int(vid))

        # normalized the intr
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = 0.5
        saved_cy = 0.5
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        w2c = world2cams[vid]
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":
    # we only use DTU for testing, not for training
    for stage in ("test",):
        #insrinsics = [1200, 680, 600.0, 600.0, 599.5, 339.5, 0, 0]

        intrinsics, world2cams, cam2worlds, near_fars = build_camera_info(
            list(range(256)), INPUT_IMAGE_DIR
        )
        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = INPUT_IMAGE_DIR / 'rgb'
            num_bytes = get_size(image_dir) // 7

            # Read images and metadata.
            images = load_images(image_dir)
            print("Num Img: ", len(images))
            example = load_metadata(intrinsics, world2cams)
            print("Num Exam: ", len(example['timestamps']))
            # Merge the images into the example.
            example["images"] = [
                images[timestamp.item()] for timestamp in example["timestamps"]
            ]
            
            assert len(images) == len(example["timestamps"])

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)
