# Yet-Another-Pedestrian-Tracker
This repository contains a simple pedestrian tracker implemented as an assignment for the Computer Vision course attended at [UniPa](https://www.unipa.it/dipartimenti/ingegneria/cds/ingegneriainformatica2035/?template=responsive&pagina=insegnamento&idInsegnamento=171775&idDocente=155776&idCattedra=167762). <br/>
The tracker uses Facebook's DETR for detection and computes the cost matrix that then will be passed to the Hungarian algorithm to make associations(linear_sum_assignment). <br/>
The Cost Matrix is computed using both IoU between previous tracks and current detections and cosine similarity between features. These similarity measures are transformed into distances since we need a Cost Matrix input for the linear_sum_assignment. <br/>
To extract visual features of pedestrians it recycles DETR's CNN outputs and then computes a RoiAlign on the bounding boxes from the detection phase.

## Tracker Policy
The tracker uses several hyperparameters:

- **min_hits**: Number of consecutive frames a pedestrian must be detected to be considered a valid track.
- **max_age**: Maximum number of frames a pedestrian remains in the scene without association due to occlusion or other issues.
- **iou_cosine_threshold**: Weight parameter for balancing IoU and cosine similarity in the cost matrix.
- **cost_matrix_threshold**: Threshold for deciding when to associate a detection with a previous frame.
- **DETR confidence threshold**: Confidence level used for accepting pedestrian detections from DETR.

## Limitations
The tracker does not account for camera motion and suffers from computational slowness when processing video due to its methodology.

## Validation Approach
To validate the tracker efficiently, we varied one hyperparameter at a time while fixing others at either average or best-found values. Additionally, for faster validation and testing, we store bounding boxes and features of each frame in memory after initial evaluation ("caching" videos). This significantly speeds up subsequent evaluations.

## Performance Evaluation
Included in this repository are the best and worst performing videos based on the tracker's evaluation on the test set.

### Best Video
https://github.com/dqvid3/Yet-Another-Pedestrian-Tracker/assets/61832683/1b91401c-45e3-409a-a325-9b2244fc6b75

### Worst Video
https://github.com/dqvid3/Yet-Another-Pedestrian-Tracker/assets/61832683/ff65bc9f-736c-4c5e-b2ec-8fab1465c2c9

## Dataset
We used the MOT17 dataset to validate and test the tracker. The 'train' folder was split equally into validation and test sets for consistent evaluation.
You can download the dataset from [MOT17-site](https://motchallenge.net/data/MOT17/) down below the Download section
