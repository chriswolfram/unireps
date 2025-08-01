<h2 align="center">Layers at Similar Depths Generate Similar Activations Across LLM Architectures</h2>

<h3 align="center"><a href="https://arxiv.org/abs/2504.08775">Paper</a>
<h5 align="center">
<a href="https://christopherwolfram.com">Christopher Wolfram</a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.aaronschein.com/cv">Aaron Schein</a>
</h5>

<hr>

## Setup
### Virtual Environment
Create a virtual environment and install prerequisites with
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
(At time of writing, Torch does not support Python 3.13, so this uses Python 3.12 instead.)

### HuggingFace Authentication
Run the following to login to HuggingFace and save an authentication token to a centralized location
```bash
huggingface-cli login
```
## Scripts
- `generate_embeddings.sh`: Generate embeddings for all models and datasets.
- `generate_knn.sh`: Compute the $k$-nearest neighbors for all embedding sets.

## Output data
Output embeddings are available on HuggingFace [here](https://huggingface.co/datasets/chriswolfram/embeddings).

Output $k$-nearest neighbor sets are available on HuggingFace [here](https://huggingface.co/datasets/chriswolfram/knn).