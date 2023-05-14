import cv2, time, csv, wx 
import depthai as dai
import numpy as np
from dot import DotDetector

# Create a wx.App instance
app = wx.App()

def get_initial_dot_spacing():
    # Use wx.TextEntryDialog to get the user's input
    dialog = wx.TextEntryDialog(None, "Please enter the initial dot spacing (in):", "Initial Dot Spacing")
    if dialog.ShowModal() == wx.ID_OK:  # The user clicked OK
        initial_dot_spacing_in = float(dialog.GetValue())  # Convert the input to a float
        return initial_dot_spacing_in
    else:  # The user clicked Cancel or closed the dialog
        app.Exit()  # Exit the program

initial_dot_spacing_in = get_initial_dot_spacing()

# Create an instance of DotDetector
dot_detector = DotDetector("moments")

def record_video(video_writer, frame):
    video_writer.write(frame)

def save_csv_data(csv_writer, timestamp, distance_in): # csv_writer, timestamp, and distance_in are parameters
    csv_writer.writerow([timestamp, distance_in])

def draw_rounded_rect(image, top_left, bottom_right, color, thickness, r):
    x1,y1 = top_left
    x2,y2 = bottom_right
    cv2.rectangle(image, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(image, (x1, y1+r), (x2, y2-r), color, -1)
    cv2.ellipse(image, (x1+r, y1+r), (r,r), 180, 0, 90, color, -1)
    cv2.ellipse(image, (x2-r, y1+r), (r,r), 270, 0, 90, color, -1)
    cv2.ellipse(image, (x1+r, y2-r), (r,r), 90, 0, 90, color, -1)
    cv2.ellipse(image, (x2-r, y2-r), (r,r), 0, 0, 90, color, -1)

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

    # Draw a circle and a single point dot at the center for each detected red dot
    for dot in red_dots:
        # Draw a circle
        cv2.circle(frame, tuple(dot), radius=20, color=(0, 255, 0), thickness=2)

        # Draw a dot at the center
        cv2.drawMarker(frame, tuple(dot), color=(0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, 
                       markerSize=2, thickness=2, line_type=cv2.LINE_AA)

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
    
    # Define the top-left and bottom-right coordinates for the button
    button_top_left = (frame.shape[1] - 120, 10)  # 120 pixels from the right and 10 pixels from the top
    button_bottom_right = (frame.shape[1] - 20, 50)  # 20 pixels from the right and 50 pixels from the top
    
    draw_rounded_rect(frame, button_top_left, button_bottom_right, button_color, -1, 10)
    
    # Define the bottom-left corner of the text string in the image
    text_org = (frame.shape[1] - 105, 38)  # Adjust these values as needed

    cv2.putText(frame, "[R]ec", text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)


    if recording_video:
        # Only display the counter when recording
        elapsed_time = current_timestamp - recording_start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        cv2.putText(frame, elapsed_time_str, (int(frame.shape[1]*0.75), 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

        # If CSV recording is active, save data to CSV
        if recording_csv:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_csv_data(csv_writer, timestamp, distance_in)
            csv_status_text = "Capturing Data"
            cv2.putText(frame, csv_status_text, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
