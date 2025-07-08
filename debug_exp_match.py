import cv2
import numpy as np
import sys
import os

TEMPLATE_PATH = "images2/exp.png"
SCREENSHOT_PATH = "screen.png"  # Change if your screenshot is named differently
DEBUG_DIR = "debug_exp"

os.makedirs(DEBUG_DIR, exist_ok=True)

def debug_template(template_path):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"[ERROR] Could not load template: {template_path}")
        return None
    print(f"[INFO] Template shape: {template.shape}")
    if len(template.shape) == 3 and template.shape[2] == 4:
        alpha = template[:, :, 3]
        print(f"[INFO] Alpha channel unique values: {np.unique(alpha)}")
        cv2.imwrite(os.path.join(DEBUG_DIR, "exp_alpha.png"), alpha)
    else:
        print("[INFO] No alpha channel detected in template.")
    return template

def debug_screenshot(screenshot_path):
    screenshot = cv2.imread(screenshot_path, cv2.IMREAD_UNCHANGED)
    if screenshot is None:
        print(f"[ERROR] Could not load screenshot: {screenshot_path}")
        return None
    print(f"[INFO] Screenshot shape: {screenshot.shape}")
    return screenshot

def match_and_debug(template, screenshot, threshold=0.8):
    # Prepare template and screenshot for matching
    if len(template.shape) == 2:
        template_bgr = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    elif template.shape[2] == 4:
        template_bgr = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
    else:
        template_bgr = template
    if len(screenshot.shape) == 2:
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
    elif screenshot.shape[2] == 4:
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    else:
        screenshot_bgr = screenshot

    mask = None
    if len(template.shape) == 3 and template.shape[2] == 4:
        alpha = template[:, :, 3]
        if np.any(alpha < 255):
            mask = alpha
            print(f"[DEBUG] Using alpha channel as mask for matching.")
        else:
            print(f"[DEBUG] Alpha channel present but fully opaque (all 255). Not using as mask.")
    else:
        print(f"[DEBUG] No alpha channel for mask.")

    if mask is not None:
        result = cv2.matchTemplate(screenshot_bgr, template_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        result = cv2.matchTemplate(screenshot_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f"[RESULT] min_val={min_val:.3f}, max_val={max_val:.3f}, min_loc={min_loc}, max_loc={max_loc}, threshold={threshold}")

    # Draw rectangle on screenshot at max_loc
    annotated = screenshot_bgr.copy()
    h, w = template_bgr.shape[:2]
    cv2.rectangle(annotated, max_loc, (max_loc[0]+w, max_loc[1]+h), (0,0,255), 2)
    cv2.putText(annotated, f"max_val={max_val:.3f}", (max_loc[0], max_loc[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    out_path = os.path.join(DEBUG_DIR, "exp_match_debug.png")
    cv2.imwrite(out_path, annotated)
    print(f"[DEBUG] Annotated match image saved to {out_path}")
    if max_val >= threshold:
        print(f"[MATCH] exp.png detected at {max_loc} with score {max_val:.3f}")
    else:
        print(f"[NO MATCH] exp.png not detected above threshold.")

def main():
    template = debug_template(TEMPLATE_PATH)
    screenshot = debug_screenshot(SCREENSHOT_PATH)
    if template is not None and screenshot is not None:
        match_and_debug(template, screenshot)

if __name__ == "__main__":
    main()
