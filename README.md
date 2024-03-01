# fine-tunning
# 从模型部署到模型微调，此项目是经过训练营学习后，结合训练营项目，自我理解消化总结，以及创新型应用。小白可star/fork



大语言模型快速入门（理论学习与微调实战）

## 搭建开发环境

- Python v3.10+
- Python 环境管理 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 交互式开发环境 [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda)
- [音频处理工具包 ffmpeg](https://phoenixnap.com/kb/install-ffmpeg-ubuntu)



### 安装 Python 依赖包

请使用 `requirements.txt` 文件进行 Python 依赖包安装：

```shell
pip install -r requirements.txt
```

### 关于 GPU 驱动和 CUDA 版本

通常，GPU 驱动和 CUDA 版本都是需要满足安装的 PyTorch 和 TensorFlow 版本。

大多数新发布的大语言模型使用了较新的 PyTorch v2.0+ 版本，Pytorch 官方认为 CUDA 最低版本是 11.8 以及匹配的 GPU 驱动版本。详情见[Pytorch官方提供的 CUDA 最低版本要求回复](https://pytorch.org/get-started/pytorch-2.0/#faqs)。

简而言之，建议直接安装当前最新的 CUDA 12.3 版本，[详情见 Nvidia 官方安装包](https://developer.nvidia.com/cuda-downloads)。

安装完成后，使用 `nvidia-smi` 指令查看版本：

```shell
nvidia-smi
```
```shell       
Fri Mar  1 11:16:55 2024
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 529.08       Driver Version: 529.08       CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8     6W /  30W |      0MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```



为了使用OpenAI API，你需要从OpenAI控制台获取一个API密钥。一旦你有了密钥，你可以将其设置为环境变量：

对于基于Unix的系统（如Ubuntu或MacOS），你可以在终端中运行以下命令：

```bash
export OPENAI_API_KEY='你的-api-key'
```

对于Windows，你可以在命令提示符中使用以下命令：

```bash
set OPENAI_API_KEY=你的-api-key
```



关于requirements,大家可以视情况下载
```bash
pip install -r requirements.txt
```


祝学习进步