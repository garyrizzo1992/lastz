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

    # Save the full screenshot
    cv2.imwrite("memu_screenshot.png", img)
    print("Full screenshot saved as memu_screenshot.png")

    # Show image with coordinates on mouse move
    print("Move mouse over image to see coordinates. Press any key to continue to cropping.")
    show_image_with_coords("Crop MEmu Screenshot", img)

    # Let user crop the region interactively with live coordinates
    cropping = [False]
    ix, iy = [0], [0]
    rect = [0, 0, 0, 0]
    temp_img = [img.copy()]

    def crop_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cropping[0] = True
            ix[0], iy[0] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and cropping[0]:
            temp = img.copy()
            cv2.rectangle(temp, (ix[0], iy[0]), (x, y), (0, 255, 0), 2)
            cv2.putText(temp, f"({x},{y})", (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            temp_img[0] = temp
        elif event == cv2.EVENT_LBUTTONUP:
            cropping[0] = False
            rect[0], rect[1], rect[2], rect[3] = min(ix[0], x), min(iy[0], y), abs(x - ix[0]), abs(y - iy[0])
            temp = img.copy()
            cv2.rectangle(temp, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
            cv2.putText(temp, f"({x},{y})", (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            temp_img[0] = temp

    cv2.namedWindow("Drag to Crop")
    cv2.setMouseCallback("Drag to Crop", crop_mouse)
    print("Drag to select crop region. Release mouse to finish. Press any key to save crop.")
    while True:
        cv2.imshow("Drag to Crop", temp_img[0])
        key = cv2.waitKey(1)
        if key != -1 and rect[2] > 0 and rect[3] > 0:
            break
    cv2.destroyWindow("Drag to Crop")

    if rect[2] == 0 or rect[3] == 0:
        print("No region selected.")
        return

    x1, y1, w1, h1 = rect[0], rect[1], rect[2], rect[3]
    cropped = img[y1:y1+h1, x1:x1+w1]
    cv2.imwrite("memu_cropped.png", cropped)
    print("Cropped screenshot saved as memu_cropped.png")

if __name__ == "__main__":
    main()