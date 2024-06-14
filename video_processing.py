import pickle
import os
import time
from PIL import Image
from detr import detect, extract_roi_features
from Tracker import Tracker


def save_cache(cache_path, video_cache):
    with open(cache_path, 'wb') as f:
        pickle.dump(video_cache, f)


def load_cache(video_cache_path):
    if os.path.exists(video_cache_path):
        with open(video_cache_path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_results(tracks, output_path, frame_number):
    with open(output_path, 'a') as f:
        for track in tracks:
            if track.active:
                x1, y1, x2, y2 = track.bbox
                w = x2 - x1
                h = y2 - y1
                line = f"{frame_number},{track.id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
                f.write(line)


def process_video(detector, tracker, video_frames, output_path, video_folder, video_cache_path, threshold_confidence,
                  draw):
    if os.path.exists(output_path):
        os.remove(output_path)
    frames = len(video_frames)
    video_cache = load_cache(video_cache_path)

    start_time = time.time()
    save = False
    if len(video_cache) < frames:
        save = True
    for frame_number, frame in enumerate(video_frames):
        frame_cache_key = f"{video_folder}_{frame_number}"
        # Controlla se le features e le bounding boxes sono già state memorizzate in cache per questo frame
        if frame_cache_key in video_cache:
            det_bboxes, det_features, det_probas = video_cache[frame_cache_key]
        else:
            # Rilevamento dei pedoni
            probas, det_bboxes, conv_features = detect(detector, frame)
            det_features = extract_roi_features(conv_features, det_bboxes)
            det_probas = probas.max(-1).values

            # Salva le features e le bounding boxes in cache
            video_cache[frame_cache_key] = (det_bboxes, det_features, det_probas)
        keep = det_probas > threshold_confidence
        det_bboxes = det_bboxes[keep]
        det_features = det_features[keep]

        # Aggiorna i tracciamenti
        tracker.update_tracks(det_bboxes, det_features)
        if draw:
            tracker.draw_tracks(frame, frame_number, video_folder)
        save_results(tracker.tracks, output_path, frame_number + 1)

        # Mostra il progresso
        elapsed_time = time.time() - start_time
        completion_percentage = (frame_number + 1) / frames * 100
        print(f'\rVideo: {video_folder}, Progresso: {completion_percentage:.2f}%, Tempo trascorso: {elapsed_time:.2f}s',
              end='', flush=True)
    print(f'\nVideo: {video_folder} completato in {time.time() - start_time:.2f} secondi.')
    if save:
        save_cache(video_cache_path, video_cache)


def run_videos(detector, params, video_folders, train_folder, output_base_folder, cache_folder, draw=False):
    for video_folder in video_folders:
        output_path = os.path.join(output_base_folder, f'{video_folder}.txt')
        frame_paths = os.path.join(train_folder, video_folder, 'img1')
        frame_names = sorted(os.listdir(frame_paths))
        frames = [Image.open(os.path.join(frame_paths, frame_name)) for frame_name in frame_names]

        video_cache_path = f'{cache_folder}/{video_folder}.pkl'
        process_video(detector, Tracker(params['max_age'],
                                        params['min_hits'],
                                        params['cost_matrix_threshold'],
                                        params['iou_cosine_threshold']),
                      frames,
                      output_path, video_folder, video_cache_path,
                      params['threshold_confidence'], draw)
