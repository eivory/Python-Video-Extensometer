import cv2,  time, csv
import depthai as dai
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from dot import DotDetector

# Ask the user to enter the initial dot spacing in inches
ROOT = tk.Tk() # Create a Tkinter window
ROOT.withdraw() # Hide the Tkinter window
initial_dot_spacing_in = simpledialog.askfloat(title="Initial Dot Spacing",
                                               prompt="Please enter the initial dot spacing (in):") # Ask the user to enter the initial dot spacing in inches
ROOT.destroy() # Destroy the Tkinter window

# Create an instance of DotDetector
dot_detector = DotDetector("moments")

def record_video(video_writer, frame):
    video_writer.write(frame)

def save_csv_data(csv_writer, timestamp, distance_in): # csv_writer, timestamp, and distance_in are parameters
    csv_writer.writerow([timestamp, distance_in])

pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(1920, 1080)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

rgb_out = pipeline.createXLinkOut()
rgb_out.setStreamName("rgb")
cam_rgb.preview.link(rgb_out.input)

device = dai.Device(pipeline)

q_rgb = device.getOutputQueue("rgb")

cv2.namedWindow("Camera Extensometer")

recording_video = False
recording_csv = False
video_writer = None
csv_writer = None
csv_file = None

pixels_per_inch = None

start_time = time.time()

# Initialize the recording start time to zero
recording_start_time = 0

previous_timestamp = None

# Main loop
while True:
    # Get the current frame from the color camera
    in_rgb = q_rgb.get()
    frame = in_rgb.getCvFrame()

    # Calculate the actual FPS
    current_timestamp = time.time()
    if previous_timestamp is None:
        fps = 30  # Default FPS for the first frame
    else:
        fps = 1.0 / (current_timestamp - previous_timestamp)
    previous_timestamp = current_timestamp

    # Rotate the frame 90 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Find the red dots in the frame
    red_dots = dot_detector.find_dots(frame)

    distance_px = None
    distance_in = None
    if len(red_dots) >= 2:
        # Calculate distance in pixels
        distance_px = np.sqrt((red_dots[0][0] - red_dots[1][0]) ** 2 + (red_dots[0][1] - red_dots[1][1]) ** 2)

        # If pixels_per_inch is not yet calculated, calculate it
        if pixels_per_inch is None and distance_px is not None:
            pixels_per_inch = distance_px / initial_dot_spacing_in

        # Calculate the distance in inches
        if pixels_per_inch is not None:
            distance_in = distance_px / pixels_per_inch

        cv2.putText(frame, f"Distance: {distance_in:.2f} in", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)

    # Draw the buttons and signifiers on the frame
    button_color = (0, 255, 0) if recording_video else (0, 0, 255)
    cv2.rectangle(frame, (10, 600), (103, 630), button_color, -1)
    cv2.putText(frame, "[R]ecord", (15, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if recording_video:
        # Only display the counter when recording
        elapsed_time = current_timestamp - recording_start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        cv2.putText(frame, elapsed_time_str, (int(frame.shape[1]*0.75), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If CSV recording is active, save data to CSV
        if recording_csv:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_csv_data(csv_writer, timestamp, distance_in)
            csv_status_text = "Capturing Data"
            cv2.putText(frame, csv_status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Camera Extensometer", frame)

    key = cv2.waitKey(1)

    # Check if the user pressed the 'q' key
    if key == ord("q"):
        break

    # Check if the user pressed the 'r' key to toggle video recording
    if key == ord("r"):
        recording_video = not recording_video
        recording_csv = not recording_csv
        if recording_video:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # Use the actual FPS when creating the VideoWriter
            video_writer = cv2.VideoWriter(f"video_{timestamp}.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (1920, 1080))

            # Create CSV Writer
            csv_file = open(f"data_{timestamp}.csv", 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp", "Distance_in"])  # Write header
            recording_start_time = current_timestamp
        else:
            # Close VideoWriter and CSV Writer
            if video_writer:
                video_writer.release()
                video_writer = None

            if csv_file:
                csv_file.close()
                csv_file = None
                csv_writer = None

cv2.destroyAllWindows()
device.close()
if video_writer is not None:
    video_writer.release()
if csv_file is not None:
    csv_file.close()
