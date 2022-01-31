
## Data
The Honda Research institute data that we used in the project (pre-processed by [NeurIPS paper](https://proceedings.neurips.cc/paper/2021/hash/dce8af15f064d1accb98887a21029b08-Abstract.html)) can be found [here](https://drive.google.com/file/d/1Xi0mA_mAj9Emp8DqUX45jyLqJDDQZGZQ/view)

## Task 1: Robust Regressor
### Standard and Robust Models
- A Standard ResNet implementation code is obatined from [here](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py)
- Robust model code is obtained from [here](https://github.com/YuShen0118/SAAP_Auto-driving_Platform/tree/785f899fb3b3ad92075318f9fcb69b8e09597202)

### simCLR Model

1. To pre-train the model using simCLR: Run the [Pretrain.ipynb](simCLR/pre_train/Pre-train.ipynb) Jupyter Notebook file
2. To fine-tune the model: Go to DM_Project/simCLR/fine_tune/ directory and run
```
python3 main.py
```

## Task 2: Denoising Autoencoder
In order to run the autoencoder code, some steps need to be completed first:
1. Download the .pt files from this link: https://drive.google.com/drive/folders/1w7GfDKQ_wsClzP8fkrbpOe-Q4nRMjGaj?usp=sharing. These files go in the saved_models folder within the autoencoder folder.
2. Download the data from this link: https://drive.google.com/drive/folders/1jzoYxJuIaClt9Q_tloYxK06tuAJrYAFi?usp=sharing. These files go in the data folder within the autoencoder folder.
3. Now, you can either open the autoencoder folder within the simCLR folder with VS Code and run the main.py file or you can navigate to the folder through the terminal and run the following command:
```
python3 main.py
```
4. This will run the code for the autoencoder portion of the project.


## Testing trained models: 
To test a trained model on validation sets, run 
```
python3 fine_tune_model.py
```
