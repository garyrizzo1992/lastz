import cv2

# Load the screenshot (your current troop status)
screenshot = cv2.imread('memu_cropped.png', cv2.IMREAD_GRAYSCALE)

# Load the templates
template_1_2 = cv2.imread('images/gather/1_2.png', cv2.IMREAD_GRAYSCALE)
template_2_2 = cv2.imread('images/gather/2_2.png', cv2.IMREAD_GRAYSCALE)

# Match templates
res_1 = cv2.matchTemplate(screenshot, template_1_2, cv2.TM_CCOEFF_NORMED)
res_2 = cv2.matchTemplate(screenshot, template_2_2, cv2.TM_CCOEFF_NORMED)

# Get max match values
max_val_1 = res_1.max()
max_val_2 = res_2.max()

print(f"Match 1/2: {max_val_1:.3f}")
print(f"Match 2/2: {max_val_2:.3f}")

# Set confidence threshold
threshold = 0.9

# Decide based on highest match
if max_val_1 >= threshold and max_val_1 > max_val_2:
    print("✅ Detected 1/2 — send a troop")
elif max_val_2 >= threshold and max_val_2 > max_val_1:
    print("❌ Detected 2/2 — all troops busy")
else:
    print("⚠️ Unable to confidently detect status")