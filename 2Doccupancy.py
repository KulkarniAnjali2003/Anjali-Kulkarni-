import cv2
import numpy as np
# Grid parameters
grid_resolution = 0.1  # meters per grid cell
grid_width = 10  # width of the room in meters
grid_height = 10  # height of the room in meters

# Initialize grid map
grid_map = np.zeros((int(grid_height / grid_resolution), int(grid_width / grid_resolution)), dtype=np.uint8)

# Background subtraction parameters
fgbg = cv2.createBackgroundSubtractorMOG2()
# Simulated camera feed (replace with actual camera feed acquisition)
def get_camera_frame():
    # Simulated image of the room (example)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Example dimensions
    # Simulate objects or people in the room for testing
    # Example: draw a person at a specific location
    cv2.rectangle(frame, (300, 200), (340, 240), (255, 255, 255), -1)  # Example person location
    return frame
while True:
    # Capture frame from camera (replace with actual camera capture code)
    frame = get_camera_frame()

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Object detection and processing
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate center of the object (for simplicity, just use the center of the bounding box)
        obj_center_x = x + w // 2
        obj_center_y = y + h // 2

        # Convert object center to grid coordinates
        grid_x = int(obj_center_x / grid_resolution)
        grid_y = int(obj_center_y / grid_resolution)

        # Update occupancy grid map
        grid_map[grid_y, grid_x] = 1  # Mark cell as occupied

    # Display occupancy grid map (for visualization purposes)
    cv2.imshow('Occupancy Grid Map', grid_map * 255)  # Scale to 0-255 for display

    # Exit on ESC key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cv2.destroyAllWindows()
