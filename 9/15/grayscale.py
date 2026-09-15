import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter


def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def to_threshold(img: Image.Image, threshold: int = 128) -> Image.Image:
    return img.convert("L").point(lambda p: 255 if p >= threshold else 0)


def erode(img: Image.Image, size: int = 3) -> Image.Image:
    return img.filter(ImageFilter.MinFilter(size))


def dilate(img: Image.Image, size: int = 3) -> Image.Image:
    return img.filter(ImageFilter.MaxFilter(size))


def subtract(img: Image.Image, other: Image.Image) -> Image.Image:
    return ImageChops.subtract(img, other)


def save_with_suffix(img: Image.Image, input_path: str, suffix: str) -> str:
    src = Path(input_path)
    output_path = str(src.with_stem(src.stem + suffix))
    img.save(output_path)
    return output_path


def kernel_matrix_text(size: int, op: str) -> str:
    row = " ".join(["1"] * size)
    matrix = "\n".join([row] * size)
    formula = {
        "erode": f"output(x,y) = min over {size}x{size} neighborhood",
        "dilate": f"output(x,y) = max over {size}x{size} neighborhood",
    }[op]
    return f"{formula}\n\nstructuring element:\n{matrix}"


def launch_gui(input_path: str) -> None:
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk

    THUMB_SIZE = (200, 200)

    root = tk.Tk()
    root.title("Kernel Playground")

    gray_img = to_grayscale(Image.open(input_path))

    threshold_var = tk.IntVar(value=128)
    erode_var = tk.IntVar(value=3)
    dilate_var = tk.IntVar(value=3)

    photo_refs: dict[str, "ImageTk.PhotoImage"] = {}
    stage_images: dict[str, Image.Image] = {}
    stage_labels: dict[str, ttk.Label] = {}
    stage_math: dict[str, ttk.Label] = {}

    def make_thumb(img: Image.Image) -> "ImageTk.PhotoImage":
        thumb = img.copy()
        thumb.thumbnail(THUMB_SIZE)
        return ImageTk.PhotoImage(thumb)

    def render(*_args) -> None:
        threshold = threshold_var.get()
        erode_size = erode_var.get()
        dilate_size = dilate_var.get()

        threshold_img = to_threshold(gray_img, threshold)
        eroded_img = erode(threshold_img, erode_size)
        dilated_img = dilate(eroded_img, dilate_size)
        subtracted_img = subtract(threshold_img, dilated_img)

        stage_images["grayscale"] = gray_img
        stage_images["threshold"] = threshold_img
        stage_images["eroded"] = eroded_img
        stage_images["dilated"] = dilated_img
        stage_images["subtracted"] = subtracted_img

        for name, img in stage_images.items():
            photo = make_thumb(img)
            photo_refs[name] = photo
            stage_labels[name].configure(image=photo)

        stage_math["grayscale"].configure(text="output(x,y) = luminance(I(x,y))")
        stage_math["threshold"].configure(
            text=f"output(x,y) = 255 if I(x,y) >= {threshold} else 0"
        )
        stage_math["eroded"].configure(text=kernel_matrix_text(erode_size, "erode"))
        stage_math["dilated"].configure(text=kernel_matrix_text(dilate_size, "dilate"))
        stage_math["subtracted"].configure(text="output = threshold_img - dilated_img")

    titles = {
        "grayscale": "Grayscale",
        "threshold": "Threshold",
        "eroded": "Eroded",
        "dilated": "Dilated",
        "subtracted": "Subtracted",
    }

    stages_frame = ttk.Frame(root, padding=10)
    stages_frame.grid(row=0, column=0)

    for col, name in enumerate(titles):
        panel = ttk.Frame(stages_frame, padding=5, relief="groove")
        panel.grid(row=0, column=col, sticky="n")

        ttk.Label(panel, text=titles[name], font=("Helvetica", 12, "bold")).pack()

        img_label = ttk.Label(panel)
        img_label.pack(pady=5)
        stage_labels[name] = img_label

        math_label = ttk.Label(panel, justify="left", font=("Menlo", 10), wraplength=180)
        math_label.pack(pady=5)
        stage_math[name] = math_label

    controls = ttk.Frame(root, padding=10)
    controls.grid(row=1, column=0, sticky="ew")

    def add_slider(label_text: str, var: tk.IntVar, frm: int, to: int, resolution: int, col: int) -> None:
        ttk.Label(controls, text=label_text).grid(row=0, column=col, sticky="w", padx=10)
        slider = tk.Scale(
            controls,
            from_=frm,
            to=to,
            resolution=resolution,
            orient="horizontal",
            variable=var,
            command=lambda _v: render(),
            length=200,
        )
        slider.grid(row=1, column=col, padx=10)

    add_slider("Threshold", threshold_var, 0, 255, 1, 0)
    add_slider("Erode kernel size", erode_var, 1, 15, 2, 1)
    add_slider("Dilate kernel size", dilate_var, 1, 15, 2, 2)

    def save_all() -> None:
        for name, img in stage_images.items():
            path = save_with_suffix(img, input_path, f"_{name}")
            print(f"Saved {name} image to {path}")

    ttk.Button(root, text="Save All Outputs", command=save_all).grid(row=2, column=0, pady=10)

    render()
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grayscale.py <input_image> [threshold] [kernel_size]")
        print("       python grayscale.py <input_image> --gui")
        sys.exit(1)

    input_path = sys.argv[1]

    if "--gui" in sys.argv:
        launch_gui(input_path)
        sys.exit(0)

    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    kernel_size = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    original = Image.open(input_path)

    gray_img = to_grayscale(original)
    gray_path = save_with_suffix(gray_img, input_path, "_grayscale")
    print(f"Saved grayscale image to {gray_path}")

    threshold_img = to_threshold(gray_img, threshold)
    threshold_path = save_with_suffix(threshold_img, input_path, "_threshold")
    print(f"Saved threshold image to {threshold_path}")

    eroded_img = erode(threshold_img, kernel_size)
    eroded_path = save_with_suffix(eroded_img, input_path, "_eroded")
    print(f"Saved eroded image to {eroded_path}")

    dilated_img = dilate(eroded_img, kernel_size)
    dilated_path = save_with_suffix(dilated_img, input_path, "_dilated")
    print(f"Saved dilated image to {dilated_path}")

    subtracted_img = subtract(threshold_img, dilated_img)
    subtracted_path = save_with_suffix(subtracted_img, input_path, "_subtracted")
    print(f"Saved subtracted image to {subtracted_path}")
