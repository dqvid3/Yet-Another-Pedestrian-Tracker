import numpy as np
from scipy.optimize import linear_sum_assignment
from Track import Track
import cv2
from utils import compute_cost_matrix
import torch


class Tracker:
    def __init__(self, max_age, min_hits, cost_matrix_threshold, iou_cosine_threshold):
        self.tracks = []
        self.next_id = 1
        self.first_frame = True
        self.max_age = max_age
        self.min_hits = min_hits
        self.cost_matrix_threshold = cost_matrix_threshold
        self.iou_cosine_threshold = iou_cosine_threshold

    def add_track(self, bbox, features):
        self.tracks.append(Track(self.next_id, bbox, features, self.max_age, self.min_hits))
        self.next_id += 1

    def update_tracks(self, det_bboxes, det_features):
        self.associate_detections(det_bboxes, det_features)
        if self.first_frame:
            for track in self.tracks:
                track.activate()
            self.first_frame = False

    def associate_detections(self, det_bboxes, det_features):
        if not self.tracks:
            for bbox, feature in zip(det_bboxes, det_features):
                self.add_track(bbox, feature)
            return

        track_bboxes = torch.stack([track.bbox for track in self.tracks])
        track_features = torch.stack([track.features for track in self.tracks])
        cost_matrix = compute_cost_matrix(track_bboxes, det_bboxes, det_features, track_features,
                                          self.iou_cosine_threshold)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.cost_matrix_threshold:
                self.tracks[r].update(det_bboxes[c], det_features[c])
                assigned_tracks.add(r)

        unassigned_detections = set(range(len(det_bboxes))) - set(col_ind)
        for i in unassigned_detections:
            self.add_track(det_bboxes[i], det_features[i])

        unassigned_tracks = set(range(len(self.tracks))) - assigned_tracks
        for i in unassigned_tracks:
            self.tracks[i].miss_detection()

    def draw_tracks(self, frame, frame_number, video_folder):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        for track in self.tracks:
            if track.active:
                x1, y1, x2, y2 = map(int, track.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(track.id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Tracking', frame)
        cv2.imwrite(f'outputs/test_videos/{video_folder}/{frame_number:04d}.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        cv2.waitKey(1)
