import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from std_msgs.msg import Float32

# Define lower and upper boundaries for red, green, and yellow
lower_red = np.array([126, 0, 0])
upper_red = np.array([179, 255, 255])
lower_green = np.array([59, 53, 0])
upper_green = np.array([87, 255, 255])
lower_yellow = np.array([0, 255, 0])
upper_yellow = np.array([90, 255, 255])

colors = {
    'red': ((0, 0, 255), [lower_red, upper_red]),
    'yellow': ((0, 255, 255), [lower_yellow, upper_yellow]),
    'green': ((0, 255, 0), [lower_green, upper_green])
}

pub = None

def find_color(frame, points, max_contour_area, prev_contours):
    mask = cv.inRange(frame, points[0], points[1])
    cnts, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
    color_contours = []
    max_area = 0
    max_contour = None
    for i, c in enumerate(cnts):
        area = cv.contourArea(c)
        if area > 1500 and area < max_contour_area:
            if hierarchy[0][i][3] == -1:  # Check if contour has no parent (not enclosed)
                M = cv.moments(c)
                cx = int(M['m10'] / (M['m00'] + 1e-5))
                cy = int(M['m01'] / (M['m00'] + 1e-5))
                if not any(cv.pointPolygonTest(contour, (cx, cy), False) for contour in prev_contours):
                    if area > max_area:
                        max_area = area
                        max_contour = c
    if max_contour is not None:
        M = cv.moments(max_contour)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        color_contours.append((max_contour, cx, cy))
    return color_contours

def find_furthest_contour(contours, ref_point):
    if not contours:
        return None
    max_dist_sq = -1
    furthest_contour = None
    for contour_info in contours:
        cx, cy = contour_info[1], contour_info[2]
        dist_sq = (cx - ref_point[0])**2 + (cy - ref_point[1])**2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            furthest_contour = contour_info
    return furthest_contour

def draw_midway_line(frame, yellow_contour_info, red_contours, green_contours):
    if yellow_contour_info:  # If yellow contour is detected
        yellow_cx, yellow_cy = yellow_contour_info[1], yellow_contour_info[2]
        furthest_contour = find_furthest_contour(red_contours + green_contours, (yellow_cx, yellow_cy))
        if furthest_contour:
            furthest_cx, furthest_cy = furthest_contour[1], furthest_contour[2]
            midway_x = (yellow_cx + furthest_cx) // 2
            midway_y = (yellow_cy + furthest_cy) // 2
            cv.line(frame, (yellow_cx, yellow_cy), (furthest_cx, furthest_cy), (0, 255, 255), 2)
            cv.circle(frame, (midway_x, midway_y), 5, (0, 255, 255), -1)
            # Get the x-coordinate relative to the center of the screen
            x_relative = midway_x - frame.shape[1] // 2
            # Map it to the range [-180, 180]
            x_mapped = (x_relative / (frame.shape[1] // 2)) * 180
            print("Midway X-coordinate:", x_mapped)
            # Publish the thrust angle
            publish_thrust_angle(x_mapped)
    else:  # If yellow contour is not detected
        if red_contours and green_contours:  # If both red and green contours are detected
            red_cx, red_cy = red_contours[0][1], red_contours[0][2]
            green_cx, green_cy = green_contours[0][1], green_contours[0][2]
            midway_x = (red_cx + green_cx) // 2
            midway_y = (red_cy + green_cy) // 2
            cv.line(frame, (red_cx, red_cy), (green_cx, green_cy), (0, 255, 255), 2)
            cv.circle(frame, (midway_x, midway_y), 5, (0, 255, 255), -1)
            # Get the x-coordinate relative to the center of the screen
            x_relative = midway_x - frame.shape[1] // 2
            # Map it to the range [-180, 180]
            x_mapped = (x_relative / (frame.shape[1] // 2)) * 180
            print("Midway X-coordinate:", x_mapped)
            # Publish the thrust angle
            publish_thrust_angle(x_mapped)

def publish_thrust_angle(x_mapped):
    global pub
    # Map the x_mapped value from the range [-180, 180] to [-1.57, 1.57]
    thrust_angle = (x_mapped / 180) * 1.57
    # Publish the thrust angle
    pub.publish(thrust_angle)

def callback(data):
    try:
        bridge = CvBridge()
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
        frame = cv.GaussianBlur(frame, (21, 21), 0)  # Apply Gaussian blur
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        color_detection_view = np.zeros_like(frame)  # Initialize color detection view
        red_contours = find_color(hsv, colors['red'][1], frame.shape[0]*frame.shape[1]/3, [])
        green_contours = find_color(hsv, colors['green'][1], frame.shape[0]*frame.shape[1]/3, [])
        yellow_contours = find_color(hsv, colors['yellow'][1], frame.shape[0]*frame.shape[1]/3, [])
        
        for name, (bgr_color, clr) in colors.items():
            color_contours = find_color(hsv, clr, frame.shape[0]*frame.shape[1]/3, [])
            for contour_info in color_contours:
                c, cx, cy = contour_info
                cv.drawContours(frame, [c], -1, bgr_color, 3)
                cv.circle(frame, (cx, cy), 7, bgr_color, -1)
                cv.putText(frame, name, (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, bgr_color, 1)
                cv.drawContours(color_detection_view, [c], -1, bgr_color, 3)  # Draw contours on color detection view
                
        draw_midway_line(frame, yellow_contours[0] if yellow_contours else None, red_contours, green_contours)
        
        resized_frame = cv.resize(frame, (960, 540))

        cv.imshow("Frame: ", resized_frame)
        cv.imshow("Color Detection View", color_detection_view)  # Show color detection view
        
        key = cv.waitKey(1)
        if key == ord('q'):
            rospy.signal_shutdown("Shutting down")
            cv.destroyAllWindows()
    except Exception as e:
        rospy.logerr("Error in processing callback: %s", str(e))

def listener():
    global pub
    # Initialize the ROS node
    rospy.init_node('image_listener', anonymous=True)
    # Create a publisher for the thrust angle
    pub = rospy.Publisher('/wamv/thrusters/middle_thrust_angle', Float32, queue_size=10)
    rospy.Subscriber("/wamv/sensors/cameras/middle_right_camera/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception occurred.")
    except Exception as e:
        rospy.logerr("An error occurred: %s", str(e))
