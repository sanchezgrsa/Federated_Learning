# Preclinical Stage Alzheimer’s Disease Detection Using MRI Scans
Implementation of IAAI-21 paper 'Preclinical Stage Alzheimer’s Disease Detection Using Magnetic Resonance Image Scans' in Pytorch

## Prerequisites
    -Python 3.7.4
    -Numpy 1.19.0
    -Pytorch 1.5.1

## Dataset
Credit Card Fraud Detection (https://www.kaggle.com/mlg-ulb/creditcardfraud/tasks?taskId=77)

## Models
Federated model with 2 users compared agains Non-Federated learning 


## Results


As our second model, we employ a transformer network for the task of preclinical AD detection. Transformer models  have  been  used  for  different  tasks  such  as  human  action recognition from videos (Girdhar et al. 2018) and text translation (Vaswani et al. 2017). Although transformer networks have been used for other tasks and applications, we firmly believe that this is the first work that employs a transformer network on MRI images of brain for preclinical stage Alzheimeir’s disease detection. Slices from a brain scan are fed to the network, and the network is expected to detect whether any sign of dementia is observable or not, even the subject is showing no signs nor symptoms of the disease yet.
![GitHub Logo](/images/transformer.png)

## Numerical Comparison
![GitHub Logo](/images/numerical_comparison.png)
