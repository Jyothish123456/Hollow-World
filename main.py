import time
import gc
import torch

from depth_map import process_images
from background import process_folder
from yolo import run_yolo_detection


def clear_gpu():
    """Free unused GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():

    print("=" * 70)
    print("        VR COLLIDER AI PIPELINE")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    total_start = time.time()

    # --------------------------------------------------
    print("\nSTEP 1 : DEPTH ANYTHING")
    # --------------------------------------------------

    start = time.time()

    process_images()

    clear_gpu()

    print(f"Finished in {time.time()-start:.2f} sec")

    # --------------------------------------------------
    print("\nSTEP 2 : BACKGROUND REMOVAL")
    # --------------------------------------------------

    start = time.time()

    process_folder()

    clear_gpu()

    print(f"Finished in {time.time()-start:.2f} sec")

    # --------------------------------------------------
    print("\nSTEP 3 : YOLO DETECTION")
    # --------------------------------------------------

    start = time.time()

    run_yolo_detection()

    clear_gpu()

    print(f"Finished in {time.time()-start:.2f} sec")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print(f"Total Time : {time.time()-total_start:.2f} seconds")


if __name__ == "__main__":
    main()