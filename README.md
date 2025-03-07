# Abstract
This code repository is related to threat detection and is used to enhance features in the original UNSW-NB15 threat detection dataset. The newly added features (referred to as Type II features) can further improve the expressive capability of traffic data, and significant performance improvements have been validated across multiple models. The enhanced training and validation data can be used to train and validate new algorithms. Here, we share our processed and enhanced UNSW-NB15 dataset, as well as the relevant feature processing and model training code. If you have any questions, please feel free to contact us.

# Dataset
## UNSW-NB15 Dataset
UNSW-NB15 public dataset.

part of important dataset is available here :
- Statistical features of 2.5 million flow data: `UNSW_NB15_x.csv` (1, 2, 3, 4)
- Statistical features of 250,000 flow data: `UNSW_NB15_trainging-set.csv`, `UNSW_NB15_testing-set.csv`

the complete data is available here:[The UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
## Deep Feature
The deep feature we extract by using proposed deep feature extraction method.
- feature_7.*.rar

`Feature Description.xlsx`: Detailed feature description document.

# Code
![Feature Extraction and Validation Framework](./extraction and validation framework.png)

## feature_extract
Data Alignment and Feature Extraction based on the original UNSW-NB15 dataset
### File and Function Description
List of Files and Their Functions:

### **Data Alignment**:
1. Data Alignment 
- step1: Process CSV
  - python file: main_format_NUSW-NB15_data.py
  - description: Clean, Transform, and Handle Special Fields
- step2: Insert Data into MangoDB
  - python file: main_NUSW-NB15_2_mongoDB.py
  - description: Store Data, aiming to improve processing performance
- step3: Parse PCAP
  - python file: main_parsing_PCAP_2_packet_data.py
  - description: Extract basic features (statistical features) based on communication content, Feature fields are shown in col_name.py
- step4: Data Alignment
  - python file: main_matching_testing_training_set.py, main_merge_1c_feature_testing_training.py, main_give_the_optimal_class_1_feature_testing_training.py
  - description: 
    1. Match Testing/Training Data to NUSW-NB15 Dataset. The matching result includes one-to-many (indices). 
    2. Reduce one-to-many features to one-to-one by randomly selecting and using minimum distance judgment.
    3. Note: At this step, the indices of the testing and training datasets corresponding to the entire UNSW-NB15 dataset are obtained.  

- step5:
  - python file: main_characteristics_1_category2.py
  - description:
    1. **make_mediacy_group_5_tuples_time**: Statistical analysis of 5-tuples, start and end times for the entire UNSW-NB15 dataset, and record the correspondence between 5-tuples and raw data.
    2. **make_mediacy_pkt_info_set**: List all statistical information extracted from packet data within the time window based on 5-tuples and times of UNSW-NB15.  
    3. **calculate_aggregate_features**: Calculate statistical features within the time window of PCAP based on the results from 2.
    4. **expand_1_category_features**: Combine Type I and Type II Features.  

## model_validation
### Feature Files
- Feature Version 7: `input/us_features/feature_7.csv`
  - Description: Type-I and Type-II Feature (i.e., using deep feature extraction method)
  - You need to extract `data/feature_7.*.rar` to `input/us_features/`
### Code:
- `algorithm/`
  - `model.py`: Entry point for code
  - `xxx.py`: Definitions for various models
- `feature_process/`
  - `featurex.py`: Processing for Feature Version X
  - `feature`: General feature processing
- `analysis/`
  - `dataset_analysis.py`: Dataset analysis, including data imbalance
  - `feature_analysis.py`: Feature analysis, including feature importance
  - `shap_analysis.py`: SHAPley method for feature importance analysis
- `utils/`
  - `calculate_utils.py`: Visualization of experimental results
  - `sample_utils.py`: Code related to partial dataset sampling
- Entry Files
  - `run.py`: Entry for data processing, training, testing, and result analysis 
    - params (example: `run.py --kfold_random_state=0 --random_state=-1 --all_count=-1 --k_fold=5 --model_name="MLP" --feature_version="feature7/raw" --oversample_all=0`)
      - model_name: Model algorithm, {RFC, MLP, KNN, LR, Efficient, Autoencoder}
      - feature_version: Feature selection, choose feature file `feature7`, where `{raw, all}` represent Feature Type 1 and Feature Type 2, respectively.
      - k_fold: Value of k for k-fold cross-validation.
      - kfold_random_state: Random seed for k-fold cross-validation.
      - random_state: Random seed for mini-batch sampling.
      - all_count: Number of samples for mini-batch sampling.
      - oversample_all: Sampling algorithm.
  - `run.sh`: Script to execute `run.py`
  - `run_param.py`: Entry for model hyperparameter tuning
- requirements.txt

### Result Records
1. Training Records: `output_us/`
   - `output_us/avg_result_record.csv`
     - Records of experimental parameters and metrics
   - `output_us/output_shap_info.csv`
     - Records of experimental parameters and SHAP storage paths
2. Model Records: `model_us/`
3. SHAPLEY Records: `output_shap/`
4. Preprocessed Data: `dataset_us/`

### References
#### Efficient-CNN-BiLSTM-for-Network-IDS
Code for Paper: Efficient-CNN-BiLSTM-for-Network-IDS  
Paper is available here: [Efficient Deep CNN-BiLSTM Model for Network Intrusion Detection | Proceedings of the 2020 3rd International Conference on Artificial Intelligence and Pattern Recognition](https://dl.acm.org/doi/10.1145/3430199.3430224)