
## Project Structure
The project consists of multiple folders, each containing specific files and functionalities:

- **DeepLabNet**: Contains scripts and pre-trained models for DeepLab.
- **MembraneSegmentation**: Contains extracted and organized files for creating datasets and training segmentation models.
- **MembraneSegmentationModify**: Improved version of the segmentation pipeline with enhanced accuracy.
- **MyNet**: Includes custom-trained models and results.
- **UNet**: Contains scripts and outputs related to UNet segmentation.
- **adjust_code**: Placeholder for additional modifications or adjustments.

## Segmentation Folder
Based on the files provided by the teacher, I extracted the parts useful for my current work and organized them into the `MembraneSegmentation` folder.

### How to Use
1. **Create Training and Validation Sets**:
   - Run `CreatDatasets.py` to generate training and validation datasets.
2. **Train Models**:
   - Run `MainDeepLab.py` or `UnetMain.py` to train the respective models.
3. **Training Accuracy**:
   - The UNet model achieves approximately **91%** accuracy on the validation set.
   - The DeepLab model achieves approximately **86%** accuracy on the validation set.
4. **Training Logs**:
   - Training logs are stored in files:
     - `Unet_train_log_batchsize_2.txt`
     - `Unet_train_log_batchsize_4.txt`
     - `deeplab_train_log_batchsize_4.txt`
   - These logs can be found in the `MembraneSegmentation` folder.
5. **Make Predictions**:
   - Run `pred_deeplab.py` or `pred_unet.py` to predict the cell membrane on the test set.
   

## SegmentationModify Folder
This folder contains improvements made to the original `MembraneSegmentation` pipeline based on the reference program. After several parameter adjustments:

- The initial validation accuracy of **81%** was improved to approximately **90%**.
- Significant enhancements were achieved in the experimental results.

### How to Use
1. **Create Training and Validation Sets**:
   - Run `CreatMyDatasets.py` to generate training and validation datasets.
2. **Train Models**:
   - Run `TrainMyNetModify.py` to train the improved models.
   - Adjust parameters to optimize validation set accuracy.
3. **Training Logs**:
   - Select logs with better validation accuracy and store them in `MyNet_train_log.txt`.
   - The file `MyNet_train_log.txt` is located in the `MembraneSegmentationModify` folder.
4. **Make Predictions**:
   - Run `predMyNet.py` to predict the cell membrane on the test set.

Additionally, the folder contains corresponding EM data, membrane data, and segmentation data:
- `ac3_EM`
- `ac3_dbseg_images`
- `ac4_EM`
- `ac4_seg_daniel`

## Summary
This project showcases multiple segmentation approaches (DeepLab and UNet) for cell membrane detection. It includes:

- Original implementations.
- Improved pipelines with enhanced accuracy.
- Comprehensive training and testing instructions.

