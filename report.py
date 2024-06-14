import os
import pandas as pd


def parse_filename(filename):
    parts = filename.replace('.txt', '').split('_')
    return {
        'threshold_confidence': float(parts[-5]),
        'min_hits': int(parts[-4]),
        'max_age': int(parts[-3]),
        'iou_cosine_threshold': float(parts[-2]),
        'cost_matrix_threshold': float(parts[-1])
    }


def read_metrics(file_path):
    with open(file_path, 'r') as file:
        header = file.readline().strip().split()
        line = file.readline().strip()
        values = [float(value) for value in line.split()]
        return dict(zip(header, values))


def generate_table(validation_folder, test_folder):
    data = []
    best_metrics = None
    test_metrics = None

    for filename in os.listdir(validation_folder):
        if filename.startswith('pedestrian_summary'):
            params = parse_filename(filename)
            metrics = read_metrics(os.path.join(validation_folder, filename))
            data.append({**params, **metrics})
        elif filename.startswith('best_pedestrian_summary'):
            params = parse_filename(filename)
            best_metrics = read_metrics(os.path.join(validation_folder, filename))
            best_metrics.update(params)
    for filename in os.listdir(test_folder):
        if filename.startswith('test_pedestrian_summary'):
            params = parse_filename(filename)
            test_metrics = read_metrics(os.path.join(test_folder, filename))
            test_metrics.update(params)

    df = pd.DataFrame(data)
    avg_df = df.groupby(['threshold_confidence', 'min_hits', 'max_age', 'iou_cosine_threshold',
                         'cost_matrix_threshold']).mean().reset_index()
    return avg_df, best_metrics, test_metrics


# Cartella contenente i risultati della validazione
validation_folder = 'outputs/validation_outputs'
test_folder = 'outputs/test_outputs'

# Generare la tabella
avg_df, best_metrics, test_metrics = generate_table(validation_folder, test_folder)

# Salvare la tabella delle metriche medie in un file CSV
avg_df.to_csv('outputs/report/validation_results.csv', index=False)

# Stampare la tabella delle metriche medie
print("Metriche Medie per Iperparametri Variabili:")
print(avg_df)

# Creare DataFrame per le metriche migliori e di test
best_df = pd.DataFrame([best_metrics])
test_df = pd.DataFrame([test_metrics])

# Salvare le tabelle delle metriche migliori e di test in file CSV
best_df.to_csv('outputs/report/best_metrics.csv', index=False)
test_df.to_csv('outputs/report/test_metrics.csv', index=False)

# Stampare le tabelle delle metriche migliori e di test
print("\nMetriche Migliori:")
print(best_df)

print("\nMetriche di Test:")
print(test_df)
