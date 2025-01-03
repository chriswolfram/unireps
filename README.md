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