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
On the following table you can see the average metrics on the validation set as the hyperparameter values vary.
|threshold_confidence|min_hits|max_age|iou_cosine_threshold|cost_matrix_threshold|HOTA  |MOTA  |MOTP  |IDSW  |MT  |ML   |Frag  |
|--------------------|--------|-------|--------------------|---------------------|------|------|------|------|----|-----|------|
|0.5                 |3       |6      |0.55                |0.75                 |25.315|23.859|74.264|1505.0|24.0|60.0 |2494.0|
|0.6                 |3       |6      |0.55                |0.75                 |25.676|24.924|74.466|1177.0|23.0|69.0 |2169.0|
|0.7                 |1       |6      |0.55                |0.75                 |26.32 |25.072|75.454|867.0 |26.0|80.0 |1815.0|
|0.7                 |2       |6      |0.55                |0.75                 |26.317|25.399|75.55 |848.0 |26.0|81.0 |1775.0|
|0.7                 |3       |6      |0.55                |0.75                 |26.308|25.599|75.615|833.0 |24.0|81.0 |1744.0|
|0.7                 |4       |1      |0.3                 |0.75                 |21.489|25.99 |75.804|1369.0|20.0|82.0 |2093.0|
|0.7                 |4       |1      |0.42                |0.75                 |24.595|25.923|75.735|1108.0|20.0|85.0 |1970.0|
|0.7                 |4       |1      |0.55                |0.5                  |20.396|14.856|76.9  |197.0 |7.0 |147.0|664.0 |
|0.7                 |4       |1      |0.55                |0.6                  |23.906|20.814|76.204|299.0 |12.0|130.0|1007.0|
|0.7                 |4       |1      |0.55                |0.7                  |25.511|23.154|76.093|521.0 |16.0|112.0|1293.0|
|0.7                 |4       |1      |0.55                |0.75                 |26.034|26.447|76.215|764.0 |20.0|85.0 |1699.0|
|0.7                 |4       |1      |0.55                |0.8                  |26.076|27.062|75.852|1163.0|25.0|82.0 |1996.0|
|0.7                 |4       |1      |0.55                |0.9                  |24.966|28.836|76.144|1611.0|27.0|71.0 |2292.0|
|0.7                 |4       |1      |0.68                |0.75                 |26.661|23.622|76.327|477.0 |19.0|111.0|1303.0|
|0.7                 |4       |1      |0.8                 |0.75                 |26.68 |22.937|76.303|436.0 |18.0|114.0|1242.0|
|0.7                 |4       |2      |0.55                |0.75                 |26.195|26.376|76.033|777.0 |21.0|85.0 |1674.0|
|0.7                 |4       |3      |0.55                |0.75                 |26.241|26.19 |75.887|791.0 |21.0|85.0 |1699.0|
|0.7                 |4       |5      |0.55                |0.75                 |26.332|25.89 |75.759|805.0 |23.0|84.0 |1714.0|
|0.7                 |4       |6      |0.55                |0.75                 |26.311|25.733|75.667|820.0 |24.0|82.0 |1728.0|
|0.7                 |4       |10     |0.55                |0.75                 |26.281|25.164|75.488|829.0 |24.0|80.0 |1732.0|
|0.7                 |4       |15     |0.55                |0.75                 |26.152|24.298|75.357|854.0 |25.0|79.0 |1756.0|
|0.7                 |5       |6      |0.55                |0.75                 |26.263|25.713|75.708|799.0 |24.0|82.0 |1706.0|
|0.8                 |3       |6      |0.55                |0.75                 |26.168|24.407|76.19 |534.0 |20.0|101.0|1265.0|
***
On the following table you can see the best average metrics on the validation set with the best hyperparameter values.
|threshold_confidence|min_hits|max_age|iou_cosine_threshold|cost_matrix_threshold|HOTA  |MOTA  |MOTP  |IDSW  |MT  |ML   |Frag  |
|--------------------|--------|-------|--------------------|---------------------|------|------|------|------|----|-----|------|
|0.7                 |4       |1      |0.55                |1.0                  |23.238|29.277|76.147|1657.0|27.0|70.0 |2321.0|
***
On the following table you can see the average metrics on the test set with the best hyperparameter values.
|threshold_confidence|min_hits|max_age|iou_cosine_threshold|cost_matrix_threshold|HOTA  |MOTA  |MOTP  |IDSW  |MT  |ML   |Frag  |
|--------------------|--------|-------|--------------------|---------------------|------|------|------|------|----|-----|------|
|0.7                 |4       |1      |0.55                |1.0                  |25.837|24.388|77.907|1229.0|82.0|76.0 |1165.0|
***
<img width="1568" alt="Screenshot 2024-06-14 alle 14 57 31" src="https://github.com/dqvid3/Yet-Another-Pedestrian-Tracker/assets/61832683/f499ed13-265a-435f-886f-23ce01438adb">

The best and worst performing video based on the tracker's evaluation on the test set, as you can see from the previous image, are the ones below.

### Best Video
https://github.com/dqvid3/Yet-Another-Pedestrian-Tracker/assets/61832683/1b91401c-45e3-409a-a325-9b2244fc6b75

### Worst Video
https://github.com/dqvid3/Yet-Another-Pedestrian-Tracker/assets/61832683/ff65bc9f-736c-4c5e-b2ec-8fab1465c2c9

## Dataset
We used the MOT17 dataset to validate and test the tracker. The 'train' folder was split equally into validation and test sets for consistent evaluation.
You can download the dataset from [MOT17-site](https://motchallenge.net/data/MOT17/) down below the Download section
