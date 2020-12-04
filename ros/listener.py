import rospy
import tf
import message_filters
import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import sys

from Queue import Queue
from fcn.config import cfg
from fcn.train_test import test_image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from scipy.optimize import minimize
from utils.blob import pad_im, chromatic_transform, add_noise
from geometry_msgs.msg import PoseStamped, PoseArray
from ycb_renderer import YCBRenderer
from utils.se3 import *
from utils.nms import nms

lock = threading.Lock()

class ImageListener:

    def __init__(self, network, dataset):

        self.net = network
        self.dataset = dataset
        self.cv_bridge = CvBridge()
        self.count = 0
        self.objects = []
        self.frame_names = []
        self.frame_lost = []
        self.renders = dict()
        self.num_lost = 50
        self.queue_size = 10

        # input
        self.im = None
        self.depth = None
        self.rgb_frame_id = None

        topic_prefix = '/deepim'
        suffix = '_%02d' % (cfg.instance_id)
        prefix = '%02d_' % (cfg.instance_id)
        self.suffix = suffix
        self.prefix = prefix
        self.topic_prefix = topic_prefix

        # initialize a node
        rospy.init_node('deepim_image_listener' + suffix)
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        rospy.sleep(3.0)
        self.pose_pub = rospy.Publisher('deepim_pose_image' + suffix, Image, queue_size=1)

        # create pose publisher for each known object class
        self.pubs = []
        for i in range(self.dataset.num_classes):
            if self.dataset.classes[i][3] == '_':
                cls = prefix + self.dataset.classes[i][4:]
            else:
                cls = prefix + self.dataset.classes[i]
            self.pubs.append(rospy.Publisher(topic_prefix + '/raw/objects/prior_pose/' + cls, PoseStamped, queue_size=1))

        if cfg.TEST.ROS_CAMERA == 'D435':
            # use RealSense D435
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.target_frame = 'measured/camera_color_optical_frame'
        elif cfg.TEST.ROS_CAMERA == 'Azure':             
            rgb_sub = message_filters.Subscriber('/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/rgb/camera_info', CameraInfo)
            self.target_frame = 'rgb_camera_link'
        else:
            # use kinect
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=2)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)
            self.target_frame = '%s_depth_optical_frame' % (cfg.TEST.ROS_CAMERA)

        # update camera intrinsics
        K = np.array(msg.K).reshape(3, 3)
        self.dataset._intrinsic_matrix = K
        print(self.dataset._intrinsic_matrix)

        # initialize tensors for testing
        num = dataset.num_classes
        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        input_blob_color = torch.cuda.FloatTensor(num, 6, height, width).detach()
        image_real_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
        image_tgt_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
        image_src_blob_color = torch.cuda.FloatTensor(num, 3, height, width).detach()
        input_blob_depth = torch.cuda.FloatTensor(num, 6, height, width).detach()
        image_real_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
        image_tgt_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
        image_src_blob_depth = torch.cuda.FloatTensor(num, 3, height, width).detach()
        affine_matrices = torch.cuda.FloatTensor(num, 2, 3).detach()
        zoom_factor = torch.cuda.FloatTensor(num, 4).detach()
        flow_blob = torch.cuda.FloatTensor(num, 2, height, width).detach()
        pcloud_tgt_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
        pcloud_src_cuda = torch.cuda.FloatTensor(height, width, 3).detach()
        flow_map_cuda = torch.cuda.FloatTensor(height, width, 2).detach()

        self.test_data = {'input_blob_color': input_blob_color,
                 'image_real_blob_color': image_real_blob_color,
                 'image_tgt_blob_color': image_tgt_blob_color,
                 'image_src_blob_color': image_src_blob_color,
                 'input_blob_depth': input_blob_depth,
                 'image_real_blob_depth': image_real_blob_depth,
                 'image_tgt_blob_depth': image_tgt_blob_depth,
                 'image_src_blob_depth': image_src_blob_depth,
                 'affine_matrices': affine_matrices,
                 'zoom_factor': zoom_factor,
                 'flow_blob': flow_blob,
                 'pcloud_tgt_cuda': pcloud_tgt_cuda,
                 'pcloud_src_cuda': pcloud_src_cuda,
                 'flow_map_cuda': flow_map_cuda}

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    # callback to save images
    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id


    def average_poses(self):
        num = len(self.objects)
        poses = np.zeros((num, 9), dtype=np.float32)
        flags = np.zeros((num, ), dtype=np.int32)

        for i in range(num):
            plist = list(self.objects[i]['poses'].queue)
            n = len(plist)
            quaternions = np.zeros((n, 4), dtype=np.float32)
            translations = np.zeros((n, 3), dtype=np.float32)

            for j in range(n):
                quaternions[j, :] = plist[j][2:6]
                translations[j, :] = plist[j][6:]

            poses[i, 0] = plist[0][0]
            poses[i, 1] = plist[0][1]
            poses[i, 2:6] = averageQuaternions(quaternions)
            poses[i, 6:] = np.mean(translations, axis=0)

            if self.objects[i]['detected']:
                flags[i] = 1

        return poses, flags


    # find posecnn pose estimation results
    def query_posecnn_detection(self):

        # detection information of the target object
        frame_names = []
        frame_lost = []
        rois_est = np.zeros((0, 7), dtype=np.float32)
        poses_est = np.zeros((0, 9), dtype=np.float32)

        # look for multiple object instances
        max_objects = 5
        for i in range(self.dataset.num_classes):
            # check posecnn frame

            if self.dataset.classes[i][3] == '_':
                source_frame_base = 'posecnn/' + self.prefix + self.dataset.classes[i][4:]
            else:
                source_frame_base = 'posecnn/' + self.prefix + self.dataset.classes[i]

            for object_id in range(max_objects):

                # check posecnn frame
                suffix_frame = '_%02d' % (object_id)
                source_frame = source_frame_base + suffix_frame

                try:
                    # detection
                    trans, rot = self.listener.lookupTransform(self.target_frame, source_frame + '_roi', rospy.Time(0))
                    n = trans[0]
                    secs = trans[1]
                    now = rospy.Time.now()
                    if abs(now.secs - secs) > 1.0:
                        print 'posecnn pose for %s time out %f %f' % (source_frame, now.secs, secs)
                        continue
                    roi = np.zeros((1, 7), dtype=np.float32)
                    roi[0, 0] = 0
                    roi[0, 1] = i
                    roi[0, 2] = rot[0] * n
                    roi[0, 3] = rot[1] * n
                    roi[0, 4] = rot[2] * n
                    roi[0, 5] = rot[3] * n
                    roi[0, 6] = trans[2]
                    rois_est = np.concatenate((rois_est, roi), axis=0)

                    # pose
                    trans, rot = self.listener.lookupTransform(self.target_frame, source_frame, rospy.Time(0))
                    pose = np.zeros((1, 9), dtype=np.float32)
                    pose[0, 0] = 0
                    pose[0, 1] = i
                    pose[0, 2] = rot[3]
                    pose[0, 3] = rot[0]
                    pose[0, 4] = rot[1]
                    pose[0, 5] = rot[2]
                    pose[0, 6:] = trans
                    poses_est = np.concatenate((poses_est, pose), axis=0)
                    frame_names.append(source_frame)
                    frame_lost.append(0)

                    print('find posecnn detection ' + source_frame)
                except:
                    continue

        if rois_est.shape[0] > 0:
            # non-maximum suppression within class
            index = nms(rois_est, 0.2)
            rois_est = rois_est[index, :]
            poses_est = poses_est[index, :]
            frame_names = [frame_names[i] for i in index]
            frame_lost = [frame_lost[i] for i in index]

        return frame_names, frame_lost, rois_est, poses_est


    # run deepim
    def run_network(self):

        with lock:
            if self.im is None:
                return
            im = self.im.copy()
            depth_cv = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id

        thread_name = threading.current_thread().name
        if not thread_name in self.renders:
            print(thread_name)
            self.renders[thread_name] = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=cfg.gpu_id, render_marker=False)
            self.renders[thread_name].load_objects(self.dataset.model_mesh_paths_target,
                                                   self.dataset.model_texture_paths_target,
                                                   self.dataset.model_colors_target)
            self.renders[thread_name].set_camera_default()
            self.renders[thread_name].set_light_pos([0, 0, 0])
            self.renders[thread_name].set_light_color([1, 1, 1])
            print self.dataset.model_mesh_paths_target
        cfg.renderer = self.renders[thread_name]

        # check the posecnn pose
        frame_names, frame_lost, rois_est, poses_est = self.query_posecnn_detection()

        # cannot initialize
        if len(self.objects) == 0 and poses_est.shape[0] == 0:
            return

        # initialization
        if len(self.objects) == 0:
            self.frame_names = frame_names
            self.frame_lost = frame_lost
            self.objects = []
            for i in range(poses_est.shape[0]):
                obj = {'frame_name': frame_names[i], 'poses': Queue(maxsize=self.queue_size), 'detected': True}
                obj['poses'].put(poses_est[i, :])
                self.objects.append(obj)
        else:
            # match detection and tracking (simple version)

            # for each detected objects
            flags_detection = np.zeros((len(frame_names), ), dtype=np.int32)
            flags_tracking = np.zeros((len(self.frame_names), ), dtype=np.int32)

            for i in range(len(frame_names)):
                for j in range(len(self.frame_names)):
                    if frame_names[i] == self.frame_names[j]:
                        # data associated
                        flags_detection[i] = 1
                        flags_tracking[j] = 1
                        self.objects[j]['detected'] = True
                        break

            # undetected
            index = np.where(flags_tracking == 0)[0]
            index_remove = []
            for i in range(len(index)):
                ind = index[i]
                self.frame_lost[ind] += 1
                self.objects[ind]['detected'] = False
                if self.frame_lost[ind] >= self.num_lost:
                    index_remove.append(ind)

            # remove item
            num = len(self.frame_names)
            if len(index_remove) > 0:
                self.frame_names = [self.frame_names[i] for i in range(num) if i not in index_remove]
                self.frame_lost = [self.frame_lost[i] for i in range(num) if i not in index_remove]
                self.objects = [self.objects[i] for i in range(num) if i not in index_remove]

            # add new object to track
            ind = np.where(flags_detection == 0)[0]
            if len(ind) > 0:
                for i in range(len(ind)):
                    self.frame_names.append(frame_names[ind[i]])
                    self.frame_lost.append(0)

                    obj = {'frame_name': frame_names[i], 'poses': Queue(maxsize=self.queue_size), 'detected': True}
                    obj['poses'].put(poses_est[ind[i], :])
                    self.objects.append(obj)

        if len(self.objects) == 0:
            return

        # run network
        poses, flags = self.average_poses()

        # only refine pose for detected objects
        index = np.where(flags == 1)[0]
        if len(index) == 0:
            return
        poses_input = poses[index, :]
        im_pose_color, pose_result = test_image(self.net, self.dataset, im, depth_cv, poses_input, self.test_data)

        pose_msg = self.cv_bridge.cv2_to_imgmsg(im_pose_color)
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = rgb_frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)

        points = self.dataset._points_all
        intrinsic_matrix = self.dataset._intrinsic_matrix

        # add poses to queue
        poses = pose_result['poses_est'][-1]
        for i in range(poses.shape[0]):
            ind = index[i]
            if self.objects[ind]['poses'].full():
                self.objects[ind]['poses'].get()
            self.objects[ind]['poses'].put(poses[i, :])
        poses, flags = self.average_poses()

        # poses
        for i in range(poses.shape[0]):
            cls = int(poses[i, 1])
            if cls >= 0:
                quat = [poses[i, 3], poses[i, 4], poses[i, 5], poses[i, 2]]
                name = self.frame_names[i].replace('posecnn', 'deepim')
                print self.dataset.classes[cls], name
                self.br.sendTransform(poses[i, 6:], quat, rospy.Time.now(), name, self.target_frame)

                # create pose msg
                msg = PoseStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = self.target_frame
                msg.pose.orientation.x = poses[i, 3]
                msg.pose.orientation.y = poses[i, 4]
                msg.pose.orientation.z = poses[i, 5]
                msg.pose.orientation.w = poses[i, 2]
                msg.pose.position.x = poses[i, 6]
                msg.pose.position.y = poses[i, 7]
                msg.pose.position.z = poses[i, 8]
                pub = self.pubs[cls]
                pub.publish(msg)

                #'''
                # reinitialization if necessary
                if poses_est.shape[0] > 0:

                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[cls,:,0]
                    x3d[1, :] = points[cls,:,1]
                    x3d[2, :] = points[cls,:,2]

                    # projection 1
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses[i, 2:6])
                    RT[:, 3] = poses[i, 6:]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x = np.divide(x2d[0, :], x2d[2, :])
                    y = np.divide(x2d[1, :], x2d[2, :])
                    x1 = np.min(x)
                    y1 = np.min(y)
                    x2 = np.max(x)
                    y2 = np.max(y)
                    area = (x2 - x1 + 1) * (y2 - y1 + 1)

                    # posecnn roi
                    ind = np.where(rois_est[:, 1] == cls)[0]
                    if len(ind) > 0:
                        x1_p = rois_est[ind, 2]
                        y1_p = rois_est[ind, 3]
                        x2_p = rois_est[ind, 4]
                        y2_p = rois_est[ind, 5]
                        area_p = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

                        # compute overlap
                        xx1 = np.maximum(x1, x1_p)
                        yy1 = np.maximum(y1, y1_p)
                        xx2 = np.minimum(x2, x2_p)
                        yy2 = np.minimum(y2, y2_p)

                        w = np.maximum(0.0, xx2 - xx1 + 1)
                        h = np.maximum(0.0, yy2 - yy1 + 1)
                        inter = w * h
                        overlap = inter / (area + area_p - inter)
                        max_overlap = np.max(overlap)
                        max_ind = np.argmax(overlap)

                        print('overlap with posecnn box %.2f' % (max_overlap))
                        if max_overlap < 0.4:
                            self.objects[i]['poses'].queue.clear()
                            self.objects[i]['poses'].put(poses_est[ind[max_ind], :].flatten())
                            print('===================================reinitialize=======================================')
