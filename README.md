# Adversarially-Trained Deep Nets Transfer Better

This github repo contains the code used for the "Adversarially-Trained Deep Nets Transfer Better" paper

## How to use

1. Download all source models into models directory and save with the appropriate name:
- https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0; save as 'imagenet_l2_3_0.pt'
- https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0; save as 'imagenet_linf_4.pt'
- https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0; save as 'imagenet_linf_8.pt'

2. Install all dependencies:

- pip install robustness
- conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

3. If you want to train a single model, run the train.py file. For more info on the run-time inputs, run "python train.py --help".

4. If you want to replicate the entire experiment run the tools/batch.py file. Keep in mind that this might take a considerable ammount of time since we fine-tune over 14 thousand models.

5. Find the logs including the validation accuracy in the results/logs folder. Use the log_extractor.py file to extract all your logs into a nice csv format.
