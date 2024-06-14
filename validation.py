from detr import load_detr
from utils import evaluate_performance
import time
import os
import shutil
import numpy as np
from video_processing import run_videos


def main():
    start_time = time.time()
    detector = load_detr()
    train_folder = 'MOT17/train'
    output_base_folder = 'TrackEval/data/trackers/mot_challenge/MOT17-train/Tracker/data'
    eval_output_path = 'TrackEval/data/trackers/mot_challenge/MOT17-train/Tracker/pedestrian_summary.txt'
    validation_output_folder = 'outputs/validation_outputs'
    cache_folder = 'cache'

    if os.path.isdir(validation_output_folder):
        shutil.rmtree(validation_output_folder)
    os.makedirs(validation_output_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)

    seqmap_file_path = 'TrackEval/data/gt/mot_challenge/seqmaps/MOT17-train.txt'
    video_folders = ['MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-10-DPM']

    with open(seqmap_file_path, 'w') as f:
        f.write('name\n')
        for video_folder in video_folders:
            f.write(f"{video_folder}\n")

    threshold_confidence_range = np.linspace(0.5, 0.8, 4)
    max_age_range = [1, 2, 3, 5, 10, 15]
    min_hits_range = [1, 2, 3, 4, 5]
    iou_cosine_threshold_range = np.linspace(0.3, 0.8, 5)
    cost_matrix_threshold_range = np.linspace(0.5, 1.0, 6)

    best_params = {
        'threshold_confidence': np.mean(threshold_confidence_range),
        'min_hits': int(np.mean(min_hits_range)),
        'max_age': int(np.mean(max_age_range)),
        'iou_cosine_threshold': np.mean(iou_cosine_threshold_range),
        'cost_matrix_threshold': np.mean(cost_matrix_threshold_range)
    }

    # Ciclo per ogni parametro
    for param_name in best_params:
        print(f"Valutazione del parametro: {param_name}")
        best_score = float('-inf')
        best_param_value = None
        # Ciclo per ogni valore del parametro corrente, partendo dal primo
        for param_value in eval(f"{param_name}_range"):
            # Imposta il parametro corrente al valore da testare
            best_params[param_name] = param_value
            print(f"Valutazione con {param_name} = {param_value}")
            run_videos(detector, best_params, video_folders, train_folder, output_base_folder, cache_folder)
            mota = evaluate_performance(eval_output_path)

            if mota > best_score:
                best_score = mota
                best_param_value = param_value

            # Salva il file nella cartella di output con il nome appropriato
            param_str = '_'.join([f"{v}" if isinstance(v, int) else f"{v:.2f}" for v in best_params.values()])
            summary_file_name = f"pedestrian_summary_{param_str}.txt"
            summary_file_path = os.path.join(validation_output_folder, summary_file_name)
            shutil.copyfile(eval_output_path, summary_file_path)

        best_params[param_name] = best_param_value
        print(f"Miglior valore per {param_name}: {best_param_value} con punteggio {best_score}")

    param_str = '_'.join([f"{v}" if isinstance(v, int) else f"{v:.2f}" for v in best_params.values()])
    summary_file_path = f"pedestrian_summary_{param_str}.txt"
    os.rename(f'{validation_output_folder}/{summary_file_path}',
              f'{validation_output_folder}/best_{summary_file_path}')
    print(f"Elaborazione di tutti i video completata in {time.time() - start_time:.2f} secondi.")
    print(f"Parametri migliori trovati: {best_params}")


if __name__ == "__main__":
    main()
