import torch
import subprocess
from torchvision.ops.boxes import box_iou
import os
import cv2


def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def compute_cost_matrix(track_bboxes, det_bboxes, det_features, track_features, k):
    iou_matrix = box_iou(track_bboxes, det_bboxes)
    appearance_cost_matrix = 1 - sim_matrix(track_features, det_features)
    cost_matrix = k * (1 - iou_matrix) + (1 - k) * appearance_cost_matrix
    return cost_matrix


def run_mot_challenge():
    command = [
        "python",
        "TrackEval/scripts/run_mot_challenge.py",
        "--BENCHMARK", "MOT17",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", "Tracker",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", "8"
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        print("Errore nell'esecuzione del comando:")
        print(result.stderr)
        return False
    return True


def evaluate_performance(output_path):
    mota = float('-inf')
    if run_mot_challenge():
        with open(output_path, 'r') as f:
            lines = f.read().split('\n')
            headers = lines[0].split()
            values = lines[1].split()
            mota_index = headers.index('MOTA')
            mota = float(values[mota_index])
    return mota


def read_frame_rate(seqinfo_path):
    with open(seqinfo_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('frameRate'):
                return int(line.split('=')[1].strip())
    return 30


def create_video_from_frames(frames_folder, video_folder):
    images = [img for img in os.listdir(frames_folder)]
    images.sort()

    fps = read_frame_rate(f'MOT17/train/{video_folder}/seqinfo.ini')

    frame = cv2.imread(os.path.join(frames_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{frames_folder}/{video_folder}.mp4', fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(frames_folder, image))
        video.write(frame)
        os.remove(os.path.join(frames_folder, image))

    video.release()
    cv2.destroyAllWindows()
