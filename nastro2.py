
import numpy as np 
import cv2
from matplotlib import pyplot as plt 

def show_image(image, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

img = cv2.imread('verdesep.png', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


composite = cv2.imread('Composite.png', cv2.IMREAD_UNCHANGED)
thresh_mean = cv2.adaptiveThreshold(
    gray,                     # source imkage (must be grayscale)
    255,                     # max value
    cv2.ADAPTIVE_THRESH_MEAN_C,  # thresholding method
    cv2.THRESH_BINARY,       # threshold type
    51,                      # block size (neighborhood size, must be odd)
    10                       # constant subtracted from the mean
)

blur = cv2.GaussianBlur(thresh_mean, (15,15), 10)



inverted = cv2.bitwise_not(blur)
inverted = inverted.astype(np.uint8)
kernel = np.ones((3,3), dtype=np.uint8)

erode =cv2.erode(inverted, kernel, iterations=3)

show_image(erode)
show_image(img)

#############################################################
# HASTA AQUI TODO ES PARA PROCESAR LA IMAGEN AHORA QUEREMOS BUSCAR LOS CENTROS


contours, _ = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
eroded_image_bgr = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)
show_image(eroded_image_bgr)
centroid_coor = []

box_width = 20   # width of the box
box_height = 20  # height of the box
thickness = 3    # border thickness

for c in contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid_coor.append([cX, cY])

        # Compute rectangle corners
        top_left = (cX - box_width // 2, cY - box_height // 2)
        bottom_right = (cX + box_width // 2, cY + box_height // 2)

        # Draw rectangle
        #cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), thickness)

        # Optional: draw the center
        cv2.circle(img, (cX, cY), 10, (0, 255, 0), 2)

# Add text showing the number of centroids
num_centroids = len(centroid_coor)
cv2.putText(
    img,
    f"Numero de centros: {num_centroids-1}",
    (10, 30),                     # top-left corner of the text
    cv2.FONT_HERSHEY_SIMPLEX,     # font
    1,                            # font scale
    (255, 0, 0),                  # color (blue)
    2                             # thickness
)

show_image(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
