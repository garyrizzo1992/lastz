import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import os

def find_bluestacks_window():
    """Find the Bluestacks window, activate it, and return its bounding box."""
    print("Searching for Bluestacks window...")
    for window in gw.getAllWindows():
        print(f"Found window: {window.title}")  # Print all window titles
        if 'BlueStacks' in window.title:  # Check for partial match
            print(f"Matching window found: {window.title}")
            window.activate()  # Bring the window to the foreground
            return window.left, window.top, window.width, window.height
    return None

def capture_bluestacks_window(region):
    """Capture the Bluestacks window as a screenshot, ignoring the top 10%."""
    x, y, w, h = region
    # Adjust the region to ignore the top 10% of the screen
    ignore_height = int(h * 0.1)  # Calculate 10% of the height
    y += ignore_height  # Move the top boundary down
    h -= ignore_height  # Reduce the height to exclude the ignored area
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_image_in_window(template_path, screenshot, threshold=0.7):  # Increased threshold to 0.7
    """Find a specific image in the screenshot and debug matches."""
    # Convert screenshot to grayscale
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    # Load and convert template to grayscale
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    # Perform template matching
    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Debugging: Print match confidence
    print(f"Match confidence for {template_path}: {max_val}")
    
    if max_val >= threshold:
        # Draw a rectangle around the match for debugging
        h, w = template.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        debug_image = screenshot.copy()
        cv2.rectangle(debug_image, top_left, bottom_right, (0, 255, 0), 2)
        
        # Save the debug image to a file
        debug_filename = os.path.join("debug", f"debug_{os.path.basename(template_path)}")
        cv2.imwrite(debug_filename, debug_image)
        print(f"Debug image saved: {debug_filename}")
        
        return max_loc  # Top-left corner of the match
    return None

def main():
    # Paths to the images you want to detect
    image_paths = [
        "images/rider.png", "images/assulter.png", "images/shooter.png", 
        "images/train.png", "images/exp.png", "images/food.png", 
        "images/zent.png", "images/wood.png", "images/steel.png", 
        "images/electric.png", "images/chest.png"
    ]
    interval = 10  # Check every 10 seconds
    region = find_bluestacks_window()
    if not region:
        print("Bluestacks window not found.")
        return

    # Bluestacks resolution
    bluestacks_width, bluestacks_height = 2164, 933

    print("Monitoring Bluestacks window...")
    while True:
        # Recalculate region in case the window is resized
        region = find_bluestacks_window()
        if not region:
            print("Bluestacks window not found.")
            return

        screenshot = capture_bluestacks_window(region)
        for image_path in image_paths:
            match_location = find_image_in_window(image_path, screenshot)
            if match_location:
                # Calculate the center of the match
                x, y = match_location
                ignore_height = int(region[3] * 0.1)  # Calculate 10% of the height
                scale_x = region[2] / bluestacks_width  # Scale factor for width
                scale_y = region[3] / bluestacks_height  # Scale factor for height
                center_x = region[0] + int((x + (cv2.imread(image_path).shape[1] // 2)) * scale_x)
                center_y = region[1] + ignore_height + int((y + (cv2.imread(image_path).shape[0] // 2)) * scale_y)
                # Print and click the position
                print(f"Found {image_path} at {match_location}, clicking at ({center_x}, {center_y})")
                pyautogui.click(center_x, center_y)
                time.sleep(interval)
                if image_path == "images/train.png":
                    time.sleep(interval)
                    screenshot = capture_bluestacks_window(region)
                    time.sleep(interval)
                    match_location = find_image_in_window("images/train2.png", screenshot)
                    if match_location:
                        x, y = match_location
                        center_x = region[0] + int((x + (cv2.imread("images/train2.png").shape[1] // 2)) * scale_x)
                        center_y = region[1] + ignore_height + int((y + (cv2.imread("images/train2.png").shape[0] // 2)) * scale_y)
                        pyautogui.click(center_x, center_y)

                    time.sleep(5)  # Wait for 5 seconds before the next click

                    screenshot = capture_bluestacks_window(region)
                    match_location = find_image_in_window("images/back.png", screenshot)
                    if match_location:
                        x, y = match_location
                        center_x = region[0] + int((x + (cv2.imread("images/back.png").shape[1] // 2)) * scale_x)
                        center_y = region[1] + ignore_height + int((y + (cv2.imread("images/back.png").shape[0] // 2)) * scale_y)
                        pyautogui.click(center_x, center_y)

                if image_path == "images/chest.png":
                    time.sleep(interval)
                    screenshot = capture_bluestacks_window(region)
                    time.sleep(interval)
                    match_location = find_image_in_window("images/chest_claim.png", screenshot)
                    if match_location:
                        x, y = match_location
                        center_x = region[0] + int((x + (cv2.imread("images/chest_claim.png").shape[1] // 2)) * scale_x)
                        center_y = region[1] + ignore_height + int((y + (cv2.imread("images/chest_claim.png").shape[0] // 2)) * scale_y)
                        pyautogui.click(center_x, center_y)

                    time.sleep(interval)  # Wait for 5 seconds before the next click

                    screenshot = capture_bluestacks_window(region)
                    match_location = find_image_in_window("images/chest-claim.png", screenshot)
                    if match_location:
                        x, y = match_location
                        center_x = region[0] + int((x + (cv2.imread("images/chest-claim.png").shape[1] // 2)) * scale_x)
                        center_y = region[1] + ignore_height + int((y + (cv2.imread("images/chest-claim.png").shape[0] // 2)) * scale_y)
                        pyautogui.click(center_x, center_y)

                    time.sleep(interval)

                    screenshot = capture_bluestacks_window(region)
                    match_location = find_image_in_window("images/chest-collect.png", screenshot)
                    if match_location:
                        x, y = match_location
                        center_x = region[0] + int((x + (cv2.imread("images/chest-collect.png").shape[1] // 2)) * scale_x)
                        center_y = region[1] + ignore_height + int((y + (cv2.imread("images/chest-collect.png").shape[0] // 2)) * scale_y)
                        pyautogui.click(center_x, center_y)

                    time.sleep(interval)
        time.sleep(interval)

if __name__ == "__main__":
    main()