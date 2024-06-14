from detr import load_detr
from video_processing import run_videos
import time
import os
import shutil
from utils import create_video_from_frames

def load_params_from_filename(folder, prefix):
    for filename in os.listdir(folder):
        if filename.startswith(prefix):
            parts = filename.replace('.txt', '').split('_')
            return {
                'threshold_confidence': float(parts[-5]),
                'min_hits': int(parts[-4]),
                'max_age': int(parts[-3]),
                'iou_cosine_threshold': float(parts[-2]),
                'cost_matrix_threshold': float(parts[-1])
            }
    return {}


def main():
    start_time = time.time()
    detector = load_detr()
    train_folder = 'MOT17/train'
    output_base_folder = 'TrackEval/data/trackers/mot_challenge/MOT17-train/Tracker/data'
    eval_output_path = 'TrackEval/data/trackers/mot_challenge/MOT17-train/Tracker/pedestrian_summary.txt'
    test_output_folder = 'outputs/test_outputs'
    test_videos_folder = 'outputs/test_videos'
    cache_folder = 'cache'

    if os.path.isdir(test_videos_folder):
         shutil.rmtree(test_videos_folder)
    if os.path.isdir(test_output_folder):
        shutil.rmtree(test_output_folder)
    os.makedirs(test_output_folder, exist_ok=True)
    os.makedirs(test_videos_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)

    seqmap_file_path = 'TrackEval/data/gt/mot_challenge/seqmaps/MOT17-train.txt'
    video_folders = ['MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-11-DPM', 'MOT17-13-DPM']

    with open(seqmap_file_path, 'w') as f:
        f.write('name\n')
        for video_folder in video_folders:
            f.write(f"{video_folder}\n")
            os.makedirs(os.path.join(test_videos_folder, video_folder), exist_ok=True)

    best_params = load_params_from_filename('outputs/validation_outputs', 'best_pedestrian_summary')
    print(f"Migliori parametri ottenuti dal validation: {best_params}")

    run_videos(detector, best_params, video_folders, train_folder, output_base_folder, cache_folder, draw=True)

    # Salva il file nella cartella di output con il nome appropriato
    param_str = '_'.join([f"{v}" if isinstance(v, int) else f"{v:.2f}" for v in best_params.values()])
    summary_file_name = f"test_pedestrian_summary_{param_str}.txt"
    summary_file_path = os.path.join(test_output_folder, summary_file_name)
    shutil.copyfile(eval_output_path, summary_file_path)

    print(f"Elaborazione di tutti i video completata in {time.time() - start_time:.2f} secondi.")

    for video_folder in video_folders:
        create_video_from_frames(f'{test_videos_folder}/{video_folder}',  video_folder)


if __name__ == "__main__":
    main()
