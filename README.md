# Anjali-Kulkarni-
Develop a 2D Occupancy Grid Map of a Room using Overhead Cameras
### Report on Object Detection and Occupancy Mapping Using OpenCV

#### Overview

This report provides an analysis of a Python script that uses OpenCV to perform real-time object detection and occupancy mapping based on a simulated camera feed. The script simulates the presence of objects in a room, detects them using contours derived from background subtraction, and updates an occupancy grid map to visualize their positions.

#### Script Breakdown

##### 1. **Imports and Initialization**

```python
import cv2
import numpy as np

# Grid parameters
grid_resolution = 0.1  # meters per grid cell
grid_width = 10  # width of the room in meters
grid_height = 10  # height of the room in meters

# Initialize grid map
grid_map = np.zeros((int(grid_height / grid_resolution), int(grid_width / grid_resolution)), dtype=np.uint8)
```

- **Explanation:**
  - **Imports:** The script imports necessary libraries (`cv2` for OpenCV and `numpy` for numerical operations).
  - **Grid Parameters:** Defines parameters for the occupancy grid, including `grid_resolution`, `grid_width`, and `grid_height`.
  - **Grid Initialization:** Initializes `grid_map` as a NumPy array filled with zeros to represent the occupancy status of each grid cell.

##### 2. **Background Subtraction**

```python
fgbg = cv2.createBackgroundSubtractorMOG2()
```

- **Explanation:**
  - **Background Subtraction:** `cv2.createBackgroundSubtractorMOG2()` creates a background subtractor object (`fgbg`) using the MOG2 algorithm. This technique helps identify moving objects by subtracting the static background from successive frames.

##### 3. **Simulated Camera Feed**

```python
def get_camera_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Example dimensions
    cv2.rectangle(frame, (300, 200), (340, 240), (255, 255, 255), -1)  # Example person location
    return frame
```

- **Explanation:**
  - **Simulated Camera Feed:** `get_camera_frame()` simulates a camera feed by generating a static image (`frame`) with a filled white rectangle representing a person at `(300, 200)` to `(340, 240)`.

##### 4. **Main Loop: Object Detection and Grid Mapping**

```python
while True:
    frame = get_camera_frame()  # Capture frame from camera

    fgmask = fgbg.apply(frame)  # Apply background subtraction to detect moving objects

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours of detected objects

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Calculate bounding box of each detected object

        obj_center_x = x + w // 2  # Calculate center of the object (bounding box)
        obj_center_y = y + h // 2

        grid_x = int(obj_center_x / grid_resolution)  # Convert object center coordinates to grid coordinates
        grid_y = int(obj_center_y / grid_resolution)

        grid_map[grid_y, grid_x] = 1  # Update occupancy grid map (mark cell as occupied)

    cv2.imshow('Occupancy Grid Map', grid_map * 255)  # Display occupancy grid map (scaled for visualization)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key press
        break
```

- **Explanation:**
  - **Frame Processing:** The `while` loop continuously captures frames from the simulated camera feed (`get_camera_frame()`), applies background subtraction (`fgbg.apply(frame)`) to detect moving objects, and finds contours (`cv2.findContours()`) of these objects.
  - **Object Detection:** For each contour found, it calculates the bounding rectangle (`x, y, w, h`), determines the center of the object, converts this center to grid coordinates (`grid_x`, `grid_y`), and updates the `grid_map` to mark the corresponding grid cell as occupied (`1`).
  - **Visualization:** Displays the occupancy grid map using `cv2.imshow()`, scaled for visualization purposes (`grid_map * 255`).
  - **Loop Termination:** The loop terminates upon pressing the ESC key (`27` in ASCII) and releases resources using `cv2.destroyAllWindows()`.

#### Recommendations for Improvement

1. **Integration with Real Camera Feed:** Replace `get_camera_frame()` with actual camera capture code using `cv2.VideoCapture()` for real-time applications.
   
2. **Performance Optimization:** Consider optimizing the script for better performance, especially with larger frame sizes or higher frame rates, possibly by implementing multi-threading or GPU acceleration.

3. **Enhanced Object Detection:** Explore advanced object detection techniques like deep learning-based models (e.g., YOLO, SSD) for improved accuracy and robustness in object detection.

4. **Grid Resolution Adjustments:** Adjust `grid_resolution`, `grid_width`, and `grid_height` parameters based on specific room dimensions and resolution requirements.

5. **Visualization Enhancements:** Improve visualization techniques for the occupancy grid map, such as using color coding or overlays to convey additional information about detected objects.

#### Conclusion

This Python script demonstrates the foundational implementation of real-time object detection and occupancy mapping using OpenCV and NumPy. It provides a starting point for applications requiring spatial awareness, such as surveillance systems, robotics, or environmental monitoring. By customizing and optimizing this script, it can be tailored to various scenarios and scaled up for more sophisticated deployments.
