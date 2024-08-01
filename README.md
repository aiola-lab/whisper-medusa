# Whisper Medusa

---------

![](assets/aiola_whisper_medusa.png)


---------


## Installation
Start with creating a virtual environment and activating it:
```bash
conda create -n whisper-medusa python=3.11 -y
conda activate whisper-medusa
```

Then install the package:
```bash
git clone https://github.com/aiola-lab/whisper-medusa.git
cd whisper-medusa
pip install -e .
```

--------
## Usage
Inference can be done using the following code:
```python
from whisper_medusa import WhisperMedusa
model = WhisperMedusa.from_pretrained("aiola/whisper-medusa")
model_output = model.generate(
    input_features,
    language=language,
)
predict_ids = model_output[0]
```

--------
## Model evaluation
To evaluate the model we assume a csv file with the following columns:
- `audio`: path to the audio file.
- `sentence`: the corresponding transcript.
- `language`: the language of the audio file.

Then run the following command:

```bash
python whisper_medusa/eval.py \
--model-name /path/to/model \
--data-path /path/to/data \
--out-file-path /path/to/output \
--language en
```

arguments description:
- `model-name`: path to local model / huggingface hub.
- `data-path`: path to the data.
- `out-file-path`: path to the output file.
- `language`: default language fallback.

-------

### Citations
- `whisper-medusa` is based on [Medusa fast decoding](https://github.com/FasterDecoding/Medusa).
```bibtex
@article{cai2024medusa,
  title={Medusa: Simple llm inference acceleration framework with multiple decoding heads},
  author={Cai, Tianle and Li, Yuhong and Geng, Zhengyang and Peng, Hongwu and Lee, Jason D and Chen, Deming and Dao, Tri},
  journal={arXiv preprint arXiv:2401.10774},
  year={2024}
}
```
