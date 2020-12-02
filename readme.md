BQAM - A repo for Benchmarking Quality Assessment Metrics 

## Installation
We recommend to run the code with virtualenv. 
The code is developed with Python 3.7.3.
Please install other prerequisites with the following command after invoking a virtual env.

```
pip install -r requirements.txt
```

## Usage
Prepare a MOS file and place it under *./mos* directory. A script **prep_live_vqa_mos.py** to process raw score files is given under *./raw/LIVE_VQA/*.
Similarly, prepare a prediction score file, rename and place it under *./prediction/DATASET/DATASET_METRIC.json*.

For example,  the following command will benchmark the *PSNR* metric on the *LIVE_VQA* dataset with *l4* function

```
python bqam.py --dataset LIVE_VQA --mos LIVE_VQA_MOS.json --prediction LIVE_VQA_PSNR.json --metric PSNR --function l4
```
