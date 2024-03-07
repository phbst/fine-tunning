模型下载

```python
import os
import subprocess

os.environ['HF_ENDPOINT'] = '[https://hf-mirror.com](https://hf-mirror.com/)'

def download_model(model_name):
try:
subprocess.run(['huggingface-cli', 'download', '--resume-download', model_name,'--local-dir',model_name], check=True)
print(f"Model '{model_name}' downloaded successfully.")
except subprocess.CalledProcessError as e:
print(f"Error downloading model '{model_name}': {e}")

def download_dataset(dataset_name):
try:
subprocess.run(['huggingface-cli', 'download', '--resume-download','--repo-type','dataset',dataset_name,"--local-dir",dataset_name], check=True)
print(f"Dataset '{dataset_name}' downloaded successfully.")
except subprocess.CalledProcessError as e:
print(f"Error downloading dataset '{dataset_name}': {e}")·

```

```python
from modelscope.models import Model
model =Model.from_pretrained('AI-ModelScope/opt-125')

from modelscope.hub.snapshot_download import snapshot_download
model= snapshot_download('AI-ModelScope/opt-125',revision='v1.0.0')

from modelscope.msdatasets import MsDataset
datasset_train=MsDataset.load('cats_and_dogs', namespace='tany0699', split='train')
```

数据查看

```python
import random
import pandas as pd
import datasets
from IPython.display import display, HTML
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
```

下载位置

```

import os

os.environ['HF_HOME'] = 'E:\\model\\HF_pipeline'
os.environ['HF_HUB_CACHE'] = 'E:\\model\\HF_pipeline\\hub'

```

计算显存

```python
# 获取当前模型占用的 GPU显存（差值为预留给 PyTorch 的显存）
memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB

print(f"{memory_footprint_mib:.2f}MiB")
```

查看精度

```python
n=0
for name, param in model.named_parameters():
    n=n+1
    print(f"Parameter {name} data type: {param.dtype}")
print(n)
```