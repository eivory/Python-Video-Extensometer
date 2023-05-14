import cv2, time, csv
import depthai as dai
import numpy as np

MIN_RED_DOT_AREA = 60 # Minimum area of a red dot in pixels

# Function to find the red dots in the frame
def find_red_dots(frame):
    red_mask = get_red_mask(frame) # Get the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contours

    red_dots = [] # Create an empty list to store the red dots
    for cnt in contours: # Iterate through the contours
        area = cv2.contourArea(cnt) # Get the area of the contour
        if area > MIN_RED_DOT_AREA: # If the area is greater than the minimum area
            M = cv2.moments(cnt) # Get the moments of the contour
            if M["m00"] != 0: # If the area is not zero
                cx = int(M["m10"] / M["m00"]) # Get the x coordinate of the center of the contour
                cy = int(M["m01"] / M["m00"]) # Get the y coordinate of the center of the contour
                red_dots.append((cx, cy)) # Add the center coordinates to the list

    return red_dots # Return the list of red dots

# Function to get the red mask
def get_red_mask(frame): 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert the frame to HSV
    lower_red1 = np.array([0, 100, 100]) # Lower HSV values for red
    upper_red1 = np.array([10, 255, 255]) # Upper HSV values for red
    lower_red2 = np.array([160, 100, 100]) # Lower HSV values for red
    upper_red2 = np.array([180, 255, 255]) # Upper HSV values for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1) # Get the mask for the first red range
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2) # Get the mask for the second red range
    red_mask = cv2.bitwise_or(mask1, mask2) # Combine the masks
    return red_mask # Return the red mask

# Function to record the video
def record_video(video_writer, frame): 
    video_writer.write(frame) # Write the frame to the video

# Function to save the data to the CSV file
def save_csv_data(csv_writer, timestamp, distance_px): 
    csv_writer.writerow([timestamp, distance_px]) 

pipeline = dai.Pipeline() # Create the camera pipeline

cam_rgb = pipeline.createColorCamera() # Create the RGB camera
cam_rgb.setPreviewSize(1920, 1080) # Set the preview size
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # Set the resolution
cam_rgb.setInterleaved(False) # Set the interleaved mode to False

rgb_out = pipeline.createXLinkOut() # Create the output stream
rgb_out.setStreamName("rgb") # Set the stream name
cam_rgb.preview.link(rgb_out.input) # Link the camera preview to the output stream

device = dai.Device(pipeline) # Create the device

q_rgb = device.getOutputQueue("rgb") # Get the output queue for the RGB stream

cv2.namedWindow("Camera Extensometer") # Create the window

recording_video = False # Initialize the recording status to False
recording_csv = False # Initialize the recording status to False
video_writer = None # Initialize the video writer to None
csv_writer = None # Initialize the CSV writer to None
csv_file = None # Initialize the CSV file to None

start_time = time.time() # Get the start time

# Initialize the recording start time to zero
recording_start_time = 0

previous_timestamp = None # Initialize the previous timestamp to None

# Main loop
while True:
    in_rgb = q_rgb.get() # Get the RGB frame
    frame = in_rgb.getCvFrame() # Get the frame as a NumPy array

    current_timestamp = time.time() # Get the current timestamp
    if previous_timestamp is None: # If the previous timestamp is None
        fps = 30 # Set the FPS to 30
    else:
        fps = 1.0 / (current_timestamp - previous_timestamp) # Calculate the FPS
    previous_timestamp = current_timestamp # Set the previous timestamp to the current timestamp

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # Rotate the frame 90 degrees counter-clockwise

    red_dots = find_red_dots(frame) # Find the red dots in the frame

    distance_px = None # Initialize the distance in pixels to None
    if len(red_dots) >= 2: # If there are at least two red dots
        cv2.circle(frame, red_dots[0], 5, (0, 255, 0), -1) # Draw a circle at the first red dot
        cv2.circle(frame, red_dots[1], 5, (0, 255, 0), -1) # Draw a circle at the second red dot

        distance = np.sqrt((red_dots[0][0] - red_dots[1][0]) ** 2 + (red_dots[0][1] - red_dots[1][1]) ** 2) # Calculate the distance between the red dots
        distance_px = distance # Set the distance in pixels to the distance

        cv2.putText(frame, f"Distance: {distance_px:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2) # Display the distance in pixels

    button_color = (0, 255, 0) if recording_video else (0, 0, 255) # Set the button color to green if recording, otherwise red
    cv2.rectangle(frame, (10, 600), (103, 630), button_color, -1) # Draw the record button
    cv2.putText(frame, "[R]ecord", (15, 622), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # Display the record button text

    # Display the time code
    elapsed_time = current_timestamp - (recording_start_time if recording_video else start_time)
    cv2.putText(frame, f"Time: {elapsed_time:.2f}s", (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if recording_video: 
        csv_status_text = "Capturing Data" # Set the onscreen status text to "Capturing Data"
        cv2.putText(frame, csv_status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Display the onscreen status text

    cv2.imshow("Camera Extensometer", frame) # Display the frame

    key = cv2.waitKey(1) # Wait for a key press

    if key == ord("q"): # If the key is "q" quit the program
        break

    if key == ord("r"): # If the key is "r" toggle the recording status
        recording_video = not recording_video # Toggle the recording status
        if recording_video: # If the recording status is True
            timestamp = time.strftime("%Y%m%d-%H%M%S") # Get the current timestamp
            video_writer = cv2.VideoWriter(f"video_{timestamp}.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame.shape[1], frame.shape[0])) # Create the video writer
            csv_file = open(f"data_{timestamp}.csv", "w", newline="") # Open the CSV file
            csv_writer = csv.writer(csv_file) # Create the CSV writer
            csv_writer.writerow(["timestamp", "distance_px"]) # Write the CSV header

            # Set the recording start time when the recording starts
            recording_start_time = current_timestamp 
        else:
            video_writer.release() # Release the video writer
            csv_file.close() # Close the CSV file

    if recording_video: # If the recording status is True
        if video_writer is not None: # If the video writer is not None
            record_video(video_writer, frame) # Record the frame
        if csv_writer is not None and distance_px is not None: # If the CSV writer is not None and the distance in pixels is not None
            elapsed_time = current_timestamp - recording_start_time # Calculate the elapsed time
            save_csv_data(csv_writer, elapsed_time, distance_px) # Save the data to the CSV file

cv2.destroyAllWindows() # Destroy all the windows
device.close() # Close the device
if video_writer is not None:
    video_writer.release() # Release the video writer
if csv_file is not None: # If the CSV file is not None
    csv_file.close() # Close the CSV file