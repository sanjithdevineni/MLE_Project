# MLE_Project
Machine Learning Engineering Project - Image Classifier


Instructions for running notebook on HiPerGator:

git clone https://github.com/sanjithdevineni/MLE_Project.git

if you wish to have the repo in HiPerGator.

1. Navigate to this link -> [HiPerGator Jupyter](https://ondemand.rc.ufl.edu/pun/sys/dashboard/batch_connect/sys/jupyter/session_contexts/new)
2. Make sure to login with your UF Account
3. Enter these numbers (or different if preferred):
    - Additional Jupyter command arguments: leave blank
    - Environment Modules: pytorch
    - Number of CPU cores: 2
    - Maximum memory (GB): 15
    - Slurm Account: leave blank (uses your default group)
    - QOS: leave blank
    - Job Time Limit (hours): 3 — enough for 15 epochs with buffer
    - Cluster partition: hpg-turin
    - Generic Resource Request (--gres): gpu:1
    - Additional Slurm Options: leave blank
4. Launch and wait for session to begin
5. Ensure necessary files are present in directory.
    - Can use `ln -s /blue/cai6108/sdevineni ~/blue_link` if files/repo is in your /blue/cai6108 directory
6. Run notebook as you wish