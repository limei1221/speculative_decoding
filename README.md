## Speculative Decoding
Speculative decoding with KV cache using Hugging Face Transformers. This repo demonstrates:
- Greedy decoding
- Greedy decoding with KV cache
- Speculative decoding with an adaptive gamma strategy

### Quick start
```bash
pip install -r requirements.txt
python speculative_decoding.py
```

### Citation
Please cite the original paper:

```
@article{leviathan2023fast,
  title={Fast Inference from Transformers via Speculative Decoding},
  author={Leviathan, Yaniv and Kalman, Matan and Matias, Yossi},
  journal={arXiv preprint arXiv:2211.17192},
  year={2023}
}
```
