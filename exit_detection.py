#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import sensor_msgs.point_cloud2 as pc2
import tf
import torch

class ExitDetector3D:
    def __init__(self):
        # Initialize the ROS node with a unique name
        rospy.init_node("exit_detector3D", anonymous=True)

        # Topics to subscribe to
        self.image_topic = "/camera/image_raw"        # Camera images
        self.pc_topic = "/orb_slam3/all_points"      # 3D map points from ORB-SLAM
        self.pose_topic = "/orb_slam3/camera_pose"   # Camera pose in world frame

        self.bridge = CvBridge()  # Bridge to convert ROS Image messages to OpenCV
        self.map_points = []      # Will store 3D map points
        self.cam_pose = None      # Will store camera pose as 4x4 matrix

        # Load YOLOv8 model for exit/door detection
        self.model = YOLO("/home/noni/Lab/yolo_learn/runs/detect/train3/weights/best.pt")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use GPU if available
        self.model.to(device)

        # Camera intrinsic parameters (from calibration)
        fx, fy, cx, cy = 458.654, 457.296, 367.215, 248.375
        self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])  # Intrinsic matrix

        # ROS Subscribers
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.pc_topic, PointCloud2, self.pc_callback)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)

        rospy.loginfo("ExitDetector3D node initialized")
        rospy.spin()  # Keep node running

    def pc_callback(self, msg):
        """
        Callback for 3D point cloud updates.
        Reads PointCloud2 message and converts it to Nx3 numpy array.
        """
        points = [p for p in pc2.read_points(msg, skip_nans=True, field_names=("x","y","z"))]
        self.map_points = np.array(points)

    def pose_callback(self, msg):
        """
        Callback for camera pose updates.
        Converts ROS PoseStamped message to 4x4 homogeneous transformation.
        """
        pos = msg.pose.position
        ori = msg.pose.orientation
        quat = [ori.x, ori.y, ori.z, ori.w]  # Quaternion
        rot = tf.transformations.quaternion_matrix(quat)  # Convert to 4x4 rotation matrix
        rot[0:3,3] = [pos.x,pos.y,pos.z]  # Set translation
        self.cam_pose = rot

    def project_points(self, points):
        """
        Projects 3D points (Nx3) from world frame into camera frame, 
        then to 2D pixel coordinates using camera intrinsics.
        Returns Nx5 array: [u, v, X_cam, Y_cam, Z_cam]
        """
        if self.cam_pose is None or len(points) == 0:
            return []

        # Downsample if point cloud is very dense
        if len(points) > 10000:
            points = points[::5]

        # Convert to homogeneous coordinates (Nx4)
        pts_hom = np.hstack((points, np.ones((len(points), 1)))).T  # 4 x N

        # Transform points into camera frame: P_cam = inv(T_wc) * P_world
        cam_pts = (np.linalg.inv(self.cam_pose) @ pts_hom).T  # N x 4

        X, Y, Z, _ = cam_pts.T
        mask = Z > 0  # Keep only points in front of camera

        # Project to 2D pixels using pinhole camera model
        u = (self.K[0,0]*X/Z + self.K[0,2]).astype(int)[mask]
        v = (self.K[1,1]*Y/Z + self.K[1,2]).astype(int)[mask]
        X, Y, Z = X[mask], Y[mask], Z[mask]

        uv = np.stack([u, v, X, Y, Z], axis=1)
        return uv

    def find_gaps(self, points_2d, image_shape):
        """
        Finds large empty regions in the projected 2D points.
        Returns bounding boxes of gaps (potential exits) as [(x1,y1,x2,y2), ...]
        """
        h, w = image_shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)

        if len(points_2d) == 0:
            return []

        points_2d = np.array(points_2d)
        u = points_2d[:,0].astype(int)
        v = points_2d[:,1].astype(int)

        # Clamp points to valid image indices
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[mask]
        v = v[mask]

        empty_mask[v, u] = 255  # Mark pixels with points

        # Invert mask to highlight empty space
        gap_mask = cv2.bitwise_not(empty_mask)

        # Find contours of empty regions
        contours, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gaps = []
        for cnt in contours:
            x, y, wc, hc = cv2.boundingRect(cnt)
            if wc*hc > 5000:  # Filter small gaps (noise)
                gaps.append((x, y, x+wc, y+hc))
        return gaps

    def verify_with_geometry(self, in_box):
        """
        Verifies if candidate 3D points in a YOLO box correspond to a real exit.
        Returns (confidence boost, reason string)
        """

        pts = np.array(in_box)  # Nx3
        depths = pts[:,2]       # Z-coordinates in camera frame

        # --- Free-space check ---
        free_space = len(pts) < 200  # Few points inside → likely empty

        # --- PCA elongation check ---
        cov = np.cov(pts.T)                   # Covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov)
        dominant = eigvecs[:, np.argmax(eigvals)]
        elongated = abs(dominant[2]) > 0.7   # Axis mostly along Z → corridor/door

        # --- Floor proximity check ---
        # Project points back to 2D to check bottom pixel
        uv = [self.project_points(np.array([p]))[0] for p in pts if len(self.project_points(np.array([p]))) > 0]
        near_floor = False
        if uv:
            _, v, _, _, _ = max(uv, key=lambda x: x[1])  # Bottom-most pixel
            h = rospy.get_param("/camera_height", 480)   # Or cv_image.shape[0]
            near_floor = (v > 0.8 * h)  # Close to floor

        # --- Decision ---
        if free_space and elongated and near_floor:
            return (0.4, "geometry+PCA")   # Give confidence boost
        else:
            return (0.0, "failed checks")

    def image_callback(self, msg):
        """
        Main callback for camera images.
        Runs YOLO detection and 3D verification, then visualizes results.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        # Run YOLO exit detection
        results = self.model(cv_image)

        # Project 3D map points to 2D image
        points_2d = self.project_points(self.map_points)

        detected = False
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                # Only consider exit class (cls=0) and threshold confidence
                if int(cls) != 0 or conf < 0.4:
                    continue

                x1,y1,x2,y2 = [int(c) for c in box]
                # Collect 3D points inside the bounding box
                in_box = [p[2:] for p in points_2d if x1<=p[0]<=x2 and y1<=p[1]<=y2]

                # Verify geometry and adjust confidence
                boost, reason = self.verify_with_geometry(in_box)
                final_conf = conf + boost

                if final_conf > 0.8:
                    # Draw verified exit
                    cv2.rectangle(cv_image, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(cv_image,f"Exit ({reason})",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

                    # Log average 3D position
                    if in_box:
                        avg_pos = np.array(in_box).mean(axis=0)
                        rospy.loginfo(f"3D exit position: {avg_pos} | Verified by {reason}")
                    detected = True

        # Fallback: detect gaps if YOLO fails
        if not detected:
            gaps = self.find_gaps(points_2d, cv_image.shape)
            for x1,y1,x2,y2 in gaps:
                cv2.rectangle(cv_image, (x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(cv_image,"Potential Exit",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
                rospy.loginfo(f"Potential exit via 3D gap at approx pixels: {(x1,y1,x2,y2)}")

        # Display annotated image
        cv2.imshow("Exit Detection", cv_image)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        ExitDetector3D()
    except rospy.ROSInterruptException:
        pass
