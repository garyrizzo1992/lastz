import pygetwindow as gw
import pyautogui
import cv2
import numpy as np

def get_memu_window(title_contains="MEmu2"):
    for window in gw.getAllWindows():
        if title_contains in window.title:
            return window
    return None

def show_image_with_coords(window_name, img):
    clone = img.copy()
    coords = [0, 0]

    def mouse_move(event, x, y, flags, param):
        coords[0], coords[1] = x, y

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_move)
    while True:
        temp = clone.copy()
        # Show coordinates at the top-left and also near the cursor
        cv2.putText(temp, f"({coords[0]},{coords[1]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(temp, f"({coords[0]},{coords[1]})", (coords[0]+10, coords[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20)
        # Exit on any key press (so user can proceed to ROI selection)
        if key != -1:
            break
    cv2.destroyWindow(window_name)

def main():
    window = get_memu_window()
    if not window:
        print("No MEmu window found.")
        return

    x, y, w, h = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Show image with coordinates on mouse move
    print("Move mouse over image to see coordinates. Select ROI and press ENTER or SPACE. Press C to cancel.")
    show_image_with_coords("Crop MEmu Screenshot", img)

    # Let user crop the region interactively
    roi = cv2.selectROI("Crop MEmu Screenshot", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No region selected.")
        return

    x1, y1, w1, h1 = roi
    cropped = img[y1:y1+h1, x1:x1+w1]
    cv2.imwrite("memu_cropped.png", cropped)
    print("Cropped screenshot saved as memu_cropped.png")

if __name__ == "__main__":
    main()