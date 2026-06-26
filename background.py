import os
from rembg import remove, new_session
import onnxruntime as ort

# -------------------------
# Check GPU Provider
# -------------------------

providers = ort.get_available_providers()

print("Available ONNX Providers:")
print(providers)

if "CUDAExecutionProvider" in providers:
    print("\n✓ Background Removal using GPU\n")
    session = new_session(
        "u2net",
        providers=["CUDAExecutionProvider"]
    )
else:
    print("\n⚠ CUDA Provider not found. Using CPU.\n")
    session = new_session("u2net")

# -------------------------
# Paths
# -------------------------

input_dir = r"C:\Users\HP\OneDrive\Desktop\Hollow World\External Image Folder\depth"

output_dir = r"C:\Users\HP\OneDrive\Desktop\Hollow World\External Image Folder\background removal"

os.makedirs(output_dir, exist_ok=True)


def process_folder():

    images = [

        f for f in os.listdir(input_dir)

        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )

    ]

    print(f"Found {len(images)} images")

    for filename in images:

        input_path = os.path.join(input_dir, filename)

        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(filename)[0]}.png"
        )

        print(f"Removing Background : {filename}")

        try:

            with open(input_path, "rb") as i:

                input_data = i.read()

            output = remove(

                input_data,

                session=session

            )

            with open(output_path, "wb") as o:

                o.write(output)

            print(f"✓ Saved : {output_path}")

        except Exception as e:

            print(e)


if __name__ == "__main__":

    process_folder()