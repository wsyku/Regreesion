regression_raw = r"F:\machine_learning\model\Regression\regression_raw"
regression_preprocessed = r"F:\machine_learning\model\Regression\regression_preprocessed"
regression_results = r"F:\machine_learning\model\Regression\regression_results"



if regression_raw is None:
    print("regression_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if regression_preprocessed is None:
    print("regression_preprocessed is not defined and nnU-Net can not be used for preprocessing ")


if regression_results is None:
    print("regression_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
