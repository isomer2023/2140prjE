## L2RPN: Learn To Run Power Network
This repository contains tools and infrastructure to work with Grid2Op. Originally intended for the L2RPN competition, it can be used on its own for non-commercial research purposes.

### Setup
We recommend using a Python virtual environment when working with this repository. The following commands should be run from the working directory of the repository (i.e. the top level).
```bash
python -m venv VIRTUAL_ENV_NAME
```

Then you can activate the Virtual environment you created:
On Windows:
```bash
./VIRTUAL_ENV_NAME/Scripts/activate
```
On Linux:
```bash
source ./VIRTUAL_ENV_NAME/bin/activate
```

From there use the `requirements.txt` to install all required packages:
```bash
pip install -r requirements.txt
```

You can now run the code in the repository.
If you have a GPU, install [cuda 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) (from Nvidia) and then install torch and torchg-eometric with cuda support enabled:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```
**Note**: Double-check the version numbers (as these may change).