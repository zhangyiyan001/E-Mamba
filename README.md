## INNFUS E-Mamba
This is the official implementation for the paper "E-Mamba: Efficient Mamba network for hyperspectral and LiDAR joint classification", published in Information Fusion, 2026.
![E-Mamba](https://github.com/zhangyiyan001/E-Mamba/blob/master/framework.png)
****

## Environment

We have successfully tested the environment only on Linux. Please ensure you have the appropriate versions of PyTorch and CUDA installed that match your computational resources.

1.  **Create and activate the Conda environment:**
    ```shell
    conda create -n E-Mamba python=3.9
    conda activate E-Mamba
    ```

2.  **Install dependencies:**
    First, install PyTorch, CUDA, and cuDNN. Then, install the remaining dependencies with the following command:
    ```shell
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

3.  **Install Mamba:**
    ```shell
    cd mamba/selective_scan_mu && pip install . && cd ../../
    ```

## Usage

You can run the model on different datasets using the following commands:

**For the Houston Dataset:**
```shell
python main.py --dataset Houston --window_size 9 --gpu 0
```

**For the Trento Dataset:**
```shell
python main.py --dataset Trento --window_size 9 --gpu 0
```

**For the MUUFL Dataset:**
```shell
python main.py --dataset MUUFL --window_size 7 --gpu 0
```

## Citation

If you find our work helpful in your research, we would appreciate it if you would consider citing our paper. Your support is our greatest motivation!
```bibtex
@article{ZHANG2026103649,
  title = {E-Mamba: Efficient Mamba network for hyperspectral and LiDAR joint classification},
  journal = {Information Fusion},
  volume = {126},
  pages = {103649},
  year = {2026},
  doi = {https://doi.org/10.1016/j.inffus.2025.103649},
}
 ```

## Contact

If you have any questions or suggestions, please feel free to reach out.
Email: zhangyiyan@hhu.edu.cn

## License

This project is released under the Apache 2.0 License.

