import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# ============================================================
# FUNCIONES
# ============================================================


def overlay_image(background, overlay, x, y, alpha=1.0, scale=1.0):
    result = background.copy()

    if scale != 1.0:
        overlay = cv2.resize(
            overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    h, w = overlay.shape[:2]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
    if x + w > background.shape[1]:
        w = background.shape[1] - x
    overlay = overlay[:h, :w]

    if overlay.shape[2] == 4:
        alpha_mask = overlay[:, :, 3] / 255.0
        alpha_mask = np.dstack([alpha_mask] * 3)
        overlay_rgb = overlay[:, :, :3]
    else:
        alpha_mask = np.ones_like(overlay[:, :, :3], dtype=float)
        overlay_rgb = overlay

    roi = result[y : y + h, x : x + w]
    blended = cv2.addWeighted(
        roi.astype(float), 1 - alpha, overlay_rgb.astype(float), alpha, 0
    )
    result[y : y + h, x : x + w] = blended.astype(np.uint8)
    return result


def show_image(image, cmap="gray", title=""):
    plt.figure()
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image,
        cmap=cmap,
    )
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# PROCESAMIENTO DE UNA SOLA IMAGEN
# ============================================================


def process_image(image_path, output_dir=None, show=False):
    print(f"Procesando: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ No se pudo leer {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray).astype(np.uint8)

    thresh_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 10
    )

    blur = cv2.GaussianBlur(thresh_mean, (9, 9), 10)
    _, final_thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(final_thresh).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    erode = cv2.erode(inverted, kernel, iterations=5)

    # Buscar centros
    contours, _ = cv2.findContours(
        erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    eroded_image_bgr = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)

    centroid_coor = []
    box_width, box_height, thickness = 20, 20, 3

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid_coor.append([cX, cY])

            top_left = (cX - box_width // 2, cY - box_height // 2)
            bottom_right = (cX + box_width // 2, cY + box_height // 2)

            cv2.rectangle(
                eroded_image_bgr, top_left, bottom_right, (0, 0, 255), thickness
            )
            cv2.circle(eroded_image_bgr, (cX, cY), 3, (0, 255, 0), -1)

    num_centroids = len(centroid_coor)
    cv2.putText(
        img,
        f"Centros: {num_centroids}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    combined = overlay_image(img, erode, 0, 0, alpha=0.2)

    if show:
        show_image(combined, title=f"{os.path.basename(image_path)} (overlay)")
        show_image(eroded_image_bgr, title="Centroides detectados")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, combined)
        print(f"✅ Guardada en {out_path}")


# ============================================================
# PROCESAR TODAS LAS IMÁGENES DE UNA CARPETA
# ============================================================

input_folder = "FotosNaroe"  # Cambia esto
output_folder = "fotoSalida"  # Carpeta donde se guardan los resultados

for file in os.listdir(input_folder):
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        image_path = os.path.join(input_folder, file)
        process_image(image_path, output_dir=output_folder, show=False)
