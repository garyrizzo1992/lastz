import cv2
import numpy as np
import subprocess
import time
import os
import argparse

def adb_screenshot(device_id=None, path="/sdcard/screen.png", local_path="screen.png"):
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += ["shell", "screencap", "-p", path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, shell=True)
    cmd_pull = ["adb"]
    if device_id:
        cmd_pull += ["-s", device_id]
    cmd_pull += ["pull", path, local_path]
    subprocess.run(cmd_pull, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, shell=True)
    img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED)
    return img

def show_image_with_coords(window_name, img):
    # Always show the full image scaled to fit a 900x600 window
    win_w, win_h = 900, 600
    img_h, img_w = img.shape[:2]
    scale = min(win_w / img_w, win_h / img_h, 1.0)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)
    coords = [0, 0]
    key = -1

    def mouse_move(event, x, y, flags, param):
        coords[0], coords[1] = x, y
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_move)
    while True:
        view = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        # Map mouse coords to original image
        orig_x = int(coords[0] / scale)
        orig_y = int(coords[1] / scale)
        cv2.putText(view, f"({orig_x},{orig_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(view, f"({orig_x},{orig_y})", (coords[0]+10, coords[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow(window_name, view)
        key = cv2.waitKey(20)
        if key != -1:
            break
    cv2.destroyWindow(window_name)
    return key

def get_device_id():
    result = subprocess.run(["adb", "devices"], capture_output=True, text=True, shell=True)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and not line.startswith("List of devices"):
            parts = line.split()
            if len(parts) == 2 and parts[1] == "device":
                return parts[0]
    return None

def save_with_transparency(img, mask_color=(255, 255, 255), threshold=5):
    """Convert specified color to transparent in the image"""
    if len(img.shape) == 3 and img.shape[2] == 3:  # Add alpha channel if not present
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        img_rgba = img.copy()
    
    # Create mask for pixels close to the mask_color
    mask = np.all(abs(img_rgba[:,:,:3] - mask_color) < threshold, axis=2)
    # Set alpha to 0 for those pixels
    img_rgba[mask, 3] = 0
    return img_rgba

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Capture and crop screenshot from Android device')
    parser.add_argument('--output', '-o', default='scrcpy_cropped.png', 
                        help='Output filename (default: scrcpy_cropped.png)')
    parser.add_argument('--transparency', '-t', action='store_true',
                        help='Save with transparency (white pixels become transparent)')
    parser.add_argument('--transparency-color', '-c', nargs=3, type=int, default=[255, 255, 255],
                        help='RGB color to make transparent (default: 255 255 255 for white)')
    parser.add_argument('--threshold', type=int, default=5,
                        help='Threshold for transparency color matching (default: 5)')
    args = parser.parse_args()

    device_id = get_device_id()
    if not device_id:
        print("No Android device detected via adb.")
        return
    print(f"Using device: {device_id}")
    
    img = None
    while True:
        try:
            print("Capturing screenshot...")
            current_img = adb_screenshot(device_id)
            if current_img is None:
                print("Failed to capture screenshot. Retrying in 3 seconds...")
                time.sleep(3)
                continue
            img = current_img
            
            print("Screenshot captured. Press 'R' to refresh, any other key to continue.")
            
            # This function now returns the key pressed
            key = show_image_with_coords("Crop Device Screenshot", img)
            
            if key == ord('r') or key == ord('R'):
                continue # Loop again to refresh
            else: # Any other key proceeds to cropping
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    if img is None:
        print("Could not capture a valid screenshot.")
        return
        
    # Save the captured screenshot
    cv2.imwrite("scrcpy_screenshot.png", img)
    print("Full screenshot saved as scrcpy_screenshot.png (upright/original orientation)")
        
    # Now rotate for cropping (landscape)
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = img_rotated
    cropping = [False]
    ix, iy = [0], [0]
    rect = [0, 0, 0, 0]
    temp_img = [img.copy()]
    preview_img = [None]
    
    # Calculate mapping from rotated to original coords
    orig_h, orig_w = img.shape[:2]
    upright_w, upright_h = orig_h, orig_w  # Dimensions after rotating back
    
    def crop_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cropping[0] = True
            ix[0], iy[0] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and cropping[0]:
            temp = img.copy()
            cv2.rectangle(temp, (ix[0], iy[0]), (x, y), (0, 255, 0), 2)
            
            # Calculate mapped coordinates in original orientation
            mapped_x1 = iy[0]
            mapped_y1 = upright_w - ix[0]
            mapped_x2 = y
            mapped_y2 = upright_w - x
            
            # Display both rotated and mapped coordinates
            cv2.putText(temp, f"Rotated: ({x},{y})", (x+10, y+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(temp, f"Original: ({mapped_x2},{mapped_y2})", (x+10, y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            temp_img[0] = temp
            
            # Update preview if dragging
            if abs(x - ix[0]) > 10 and abs(y - iy[0]) > 10:  # Only if selection has meaningful size
                x1, y1 = min(ix[0], x), min(iy[0], y)
                w, h = abs(x - ix[0]), abs(y - iy[0])
                
                # Create upright version for preview
                img_upright_preview = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                upright_h_preview, upright_w_preview = img_upright_preview.shape[:2]

                # *** CORRECTED PREVIEW MAPPING ***
                crop_x_preview = y1
                crop_y_preview = upright_h_preview - (x1 + w)
                crop_w_preview = h
                crop_h_preview = w
                
                # Clamp values
                crop_x_preview = max(0, min(crop_x_preview, upright_w_preview - crop_w_preview))
                crop_y_preview = max(0, min(crop_y_preview, upright_h_preview - crop_h_preview))
                
                # Extract preview
                if crop_w_preview > 0 and crop_h_preview > 0:
                    preview = img_upright_preview[crop_y_preview:crop_y_preview+crop_h_preview, crop_x_preview:crop_x_preview+crop_w_preview].copy()
                    if preview.size > 0:
                        preview_img[0] = preview
                        cv2.imshow("Preview", preview)
            
        elif event == cv2.EVENT_LBUTTONUP:
            cropping[0] = False
            rect[0], rect[1], rect[2], rect[3] = min(ix[0], x), min(iy[0], y), abs(x - ix[0]), abs(y - iy[0])
            temp = img.copy()
            cv2.rectangle(temp, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)
            
            # Display selection information
            x1, y1, w, h = rect
            mapped_x = y1
            mapped_y = upright_w - (x1 + w)
            mapped_w = h
            mapped_h = w
            
            cv2.putText(temp, f"Selection: {w}x{h}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(temp, f"Mapped: {mapped_w}x{mapped_h}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            temp_img[0] = temp
    
    cv2.namedWindow("Drag to Crop")
    cv2.setMouseCallback("Drag to Crop", crop_mouse)
    print("Drag to select crop region. Release mouse to finish.")
    print("Press Enter to save, Esc to cancel, R to reset selection")
    
    while True:
        cv2.imshow("Drag to Crop", temp_img[0])
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to cancel
            print("Cropping canceled.")
            cv2.destroyAllWindows()
            return
        elif key == ord('r') or key == ord('R'):  # R key to reset
            rect = [0, 0, 0, 0]
            temp_img[0] = img.copy()
            if preview_img[0] is not None:
                cv2.destroyWindow("Preview")
                preview_img[0] = None
            print("Selection reset.")
        elif key == 13 and rect[2] > 0 and rect[3] > 0:  # Enter key to confirm
            break
    
    cv2.destroyAllWindows()
    
    if rect[2] == 0 or rect[3] == 0:
        print("No region selected.")
        return
        
    # Correct mapping for 90-degree clockwise rotation:
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    img_upright = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    upright_h, upright_w = img_upright.shape[:2]
    
    # *** FINAL CORRECTED MAPPING LOGIC ***
    crop_x = y
    crop_y = upright_h - (x + w)
    crop_w = h
    crop_h = w
    
    # Clamp crop_x and crop_y to be >= 0 and within bounds
    crop_x = max(0, min(crop_x, upright_w - crop_w))
    crop_y = max(0, min(crop_y, upright_h - crop_h))
    
    cropped_upright = img_upright[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # Apply transparency if requested
    if args.transparency:
        cropped_upright = save_with_transparency(
            cropped_upright, 
            mask_color=tuple(args.transparency_color), 
            threshold=args.threshold
        )
        
    cv2.imwrite(args.output, cropped_upright)
    print(f"Cropped screenshot saved as {args.output} (upright/original orientation)")
    print(f"Mapped coordinates: crop_x={crop_x}, crop_y={crop_y}, crop_w={crop_w}, crop_h={crop_h}")
    
    # Display the final cropped image
    print("Displaying final cropped image. Press any key to exit.")
    cv2.imshow("Final Cropped Image", cropped_upright)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
