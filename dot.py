import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.draw import disk
from sklearn.linear_model import LinearRegression

class DotDetector:
    def __init__(self, method="contour"):
        self.method = method

    def get_red_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        return red_mask

    def find_dots_contour(self, frame):
        red_mask = self.get_red_mask(frame)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 60:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    red_dots.append((cx, cy))

        return red_dots

    def find_dots_hough(self, frame):
        red_mask = self.get_red_mask(frame)
        circles = cv2.HoughCircles(red_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=30, minRadius=0, maxRadius=0)
        
        red_dots = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                red_dots.append((i[0], i[1]))

        return red_dots

    def find_dots_radial_symmetry(self, frame, radius=5):
        red_mask = self.get_red_mask(frame)
        y_indices, x_indices = np.indices((red_mask.shape))
        x_indices = x_indices - np.mean(x_indices[red_mask > 0])
        y_indices = y_indices - np.mean(y_indices[red_mask > 0])
        r_indices = np.hypot(x_indices, y_indices)
        sorted_indices = np.argsort(r_indices.flat)
        r_sorted = r_indices.flat[sorted_indices]
        i_sorted = red_mask.flat[sorted_indices]
        r_values = r_sorted[r_sorted < radius]
        i_values = i_sorted[r_sorted < radius]
        maxima_image = np.zeros_like(red_mask)
        maxima_image.flat[sorted_indices[:len(r_values)]] = i_values
        maxima_image = np.clip(maxima_image, 0, 255).astype(np.uint8)
        labels = label(maxima_image > 0)
        peaks = peak_local_max(maxima_image, labels=labels)

        red_dots = [(x, y) for y, x in peaks]
        return red_dots

    def find_dots_moments(self, frame):
        red_mask = self.get_red_mask(frame)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_dots = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                red_dots.append((cx, cy))
        return red_dots

    def find_dots_enclosing_circle(self, frame):
        red_mask = self.get_red_mask(frame)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_dots = []
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            red_dots.append(center)
        return red_dots

    def find_dots_least_squares(self, frame):
        red_mask = self.get_red_mask(frame)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_dots = []
        for cnt in contours:
            coords = cnt[:, 0, :]
            x = coords[:, 0].reshape(-1, 1)
            y = coords[:, 1].reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            cx = model.intercept_[0]
            cy = model.coef_[0][0] * cx
            red_dots.append((int(cx), int(cy)))
        return red_dots

    def find_dots(self, frame):
        if self.method == "contour":
            return self.find_dots_contour(frame)
        elif self.method == "hough":
            return self.find_dots_hough(frame)
        elif self.method == "radial_symmetry":
            return self.find_dots_radial_symmetry(frame)
        elif self.method == "moments":
            return self.find_dots_moments(frame)
        elif self.method == "enclosing_circle":
            return self.find_dots_enclosing_circle(frame)
        elif self.method == "least_squares":
            return self.find_dots_least_squares(frame)
        else:
            raise ValueError("Invalid method")