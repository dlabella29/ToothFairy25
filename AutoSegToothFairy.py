from monai.apps.auto3dseg import AutoRunner

def main():
   input_config = {
       "modality": "ct",
       "dataroot": "",
       "datalist": "/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/ToothFairy/ToothFairyJsonAutoSegSubmit.json",
       "sigmoid": False,
       "resample": True,
       "resample_resolution": [0.6, 0.6, 0.6],
       "num_epochs": 500,
       "intensity_bounds": [-1000, 3880],
       "num_workers": 3,
       "amp": True,
       "use_amp": True,
       "cache_rate": 0,
       "num_images_per_batch": 1,
       "learning_rate": 0.0003,
       "loss": {
        "_target_": "monai.losses.dice.DiceCELoss",
        "include_background": False,   # drop class 0 from the Dice term
        "squared_pred": True,
        "smooth_nr": 1.0e-05,
        "smooth_dr": 1.0e-05,
        "softmax": True,
        "sigmoid": False,
        "to_onehot_y": True  
        },

 }
   runner = AutoRunner(input=input_config, algos = "segresnet", work_dir= "/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/ToothFairy/AutoToothWorkDirChallenge",)
   runner.run()

if __name__ == '__main__':
  main()

#tensorboard --bind_all --logdir=/home/dlabella29/Auto3DSegDL/PythonProject/DEEP-PSMA_BaseLine/ToothFairy/AutoToothWorkDir/segresnet_0

#watch -n 1 nvidia-smi
