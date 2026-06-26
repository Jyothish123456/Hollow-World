import os
import sys
import cv2
import torch
import numpy as np

# -----------------------------
# Enable fastest CUDA settings
# -----------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

base_path = os.getcwd()

repo_path = os.path.join(base_path, "Depth-Anything-V2")

if not os.path.exists(repo_path):
    print("Depth Anything repo not found.")
    sys.exit()

sys.path.insert(0, repo_path)

from depth_anything_v2.dpt import DepthAnythingV2


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDepth Anything running on : {device}\n")

model_configs = {
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256,512,1024,1024]
    }
}

model = DepthAnythingV2(**model_configs["vitl"])

checkpoint = os.path.join(
    base_path,
    "checkpoints",
    "depth_anything_v2_vitl.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint,
        map_location=device
    )
)

model.to(device)

model.eval()


def process_images():

    input_dir = r"C:\Users\HP\OneDrive\Desktop\Hollow World\External Image Folder\sc imags"

    output_dir = r"C:\Users\HP\OneDrive\Desktop\Hollow World\External Image Folder\depth"

    os.makedirs(output_dir,exist_ok=True)

    images = [

        f

        for f in os.listdir(input_dir)

        if f.lower().endswith(
            (".png",".jpg",".jpeg")
        )

    ]

    print(f"Found {len(images)} images")

    for filename in images:

        image = cv2.imread(
            os.path.join(input_dir,filename)
        )

        if image is None:
            continue

        with torch.no_grad():

            with torch.cuda.amp.autocast():

                depth = model.infer_image(

                    image,

                    input_size=518

                )

        depth = depth.astype(np.float32)

        depth = depth.max()-depth

        depth = (

            depth-depth.min()

        )/(

            depth.max()-depth.min()

        )

        depth16 = (

            depth*65535

        ).astype(np.uint16)

        preview = (

            depth*255

        ).astype(np.uint8)

        cv2.imwrite(

            os.path.join(

                output_dir,

                f"depth_raw_{os.path.splitext(filename)[0]}.png"

            ),

            depth16

        )

        cv2.imwrite(

            os.path.join(

                output_dir,

                f"preview_{os.path.splitext(filename)[0]}.jpg"

            ),

            cv2.applyColorMap(

                preview,

                cv2.COLORMAP_MAGMA

            )

        )

        print(f"Processed {filename}")


if __name__ == "__main__":

    process_images()