### Segmentation of pertri-dish explant: Predict explant ID
+ step1: generating training data from explant callus/shoot/leaf annotation based on isolation.
+ step2: 
    * train a PSPNet with ResNet50 as backbone for explant ID segmentation
    * Refine the network segmentation result using RGB image

### packages
+ Torch 1.1.0 + torchvision 0.3.0 + python3
+ compile pymeanshift from https://github.com/fjean/pymeanshift.git

### How to prepare data
Make sure the data to be processed in level2 directory:
-- Image_Path
   -- subPath1
        -- img_1_0.jpg
        -- img_1_1.jpg
   -- subPath2
        -- img_2_0.jpg
        -- img_2_1.jpg
        -- img_2_2.jpg

### Running on new data:
1. set path of data to be processed and saving location in config.py
2. generating training examples at the begining. 
    ```python step1_generate_instance_ann.py
    ```
3. train the network.
    + cd network_segment
    + set parameters and path in config/config_plant_explant_x.json
    + train a network with: 
        ```python train.py --config config/config_plant_explant_x.json
        ``` 
    + inference on new data:
        ```python infer_explant.py --config xxx.json --model xxx.pth --images  $PATH_TO_IMAGES --output $PATH_OUTPUT
        ```
4. run step2 
    ``` cd ../
        python step2_segment_ms.py
    ```

