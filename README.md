# CS_dual_view
This repository contains the testing code for "Dual-view separation and reconstruction from one random compressed measurement".

## Environment Configuration
Before you start, please make sure the following steps are completed:
1. **Matconvnet Configuration**: Configure Matconvnet for Matlab. Follow the instructions provided in the official [Matconvnet installation guide](http://www.vlfeat.org/matconvnet/install/). After configuration, ensure that the provided `vl_simplenn.m` file is copied into the 'Matconvnet/matlab/simplenn' directory.
2. **Python Packages Installation**: Install the required Python packages. In the terminal, navigate to the directory containing the `requirements.txt` file and run the following command:
```bash
pip install -r requirements.txt
```
3. **Python Interpreter Configuration for Matlab**: Your Matlab environment should be set up with a Python interpreter. Use the following Matlab command to configure the Python interpreter: 
```bash
pyenv('Version', '/path/to/your/python/environment/python.exe')
```

## Testing with Restormer

If you wish to test this code with Restormer, please follow these steps:

1. Download the file named 'gaussian_gray_denoising_blind.pth' from the following Google Drive link: [Google Drive](https://drive.google.com/drive/folders/1rEAHUBkA9uCe9Q0AzI5zkYxePSgxYDEG).
2. After downloading, place this file into the 'CS_dual_view/Trained_Weights/Restormer/pretrained_models' directory.

By following these instructions, you should be able to test the "Dual-view separation and reconstruction from one random compressed measurement" code effectively.
