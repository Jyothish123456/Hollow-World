import os
import cv2
import torch

from ultralytics import YOLO

# ------------------------------------
# CUDA Optimizations
# ------------------------------------

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 0 if torch.cuda.is_available() else "cpu"

print()

if torch.cuda.is_available():

    print("YOLO running on GPU")
    print(torch.cuda.get_device_name(0))

else:

    print("YOLO running on CPU")

print()

# ------------------------------------

model = YOLO("yolov8s.pt")

model.to(device)

# ------------------------------------

input_dir = r"C:\Users\HP\OneDrive\Desktop\Hollow World\External Image Folder\background removal"

output_dir = r"C:\Users\HP\OneDrive\Desktop\Hollow World\External Image Folder\yolo"

os.makedirs(output_dir, exist_ok=True)


def run_yolo_detection():

    valid = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    )

    images = [

        f

        for f in os.listdir(input_dir)

        if f.lower().endswith(valid)

    ]

    print(f"Found {len(images)} images")

    for filename in images:

        path = os.path.join(input_dir, filename)

        img = cv2.imread(path)

        if img is None:

            continue

        with torch.no_grad():

            results = model(

                path,

                device=device,

                conf=0.10,

                iou=0.45,

                verbose=False

            )

        detected = False

        for result in results:

            if len(result.boxes) == 0:

                continue

            detected = True

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                label = f"{model.names[cls]} {conf:.2f}"

                cv2.rectangle(

                    img,

                    (x1, y1),

                    (x2, y2),

                    (0,255,0),

                    2

                )

                cv2.putText(

                    img,

                    label,

                    (x1,max(20,y1-10)),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (0,255,0),

                    2

                )

        if detected:

            save = os.path.join(

                output_dir,

                "detected_"+filename

            )

            cv2.imwrite(

                save,

                img

            )

            print(f"✓ Saved {filename}")

        else:

            print(f"No objects : {filename}")


if __name__ == "__main__":

    run_yolo_detection()