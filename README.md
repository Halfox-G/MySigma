# MySigma

## Evnironment Setup

### Clone the repository

```bash
git clone https://github.com/Halfox-G/MySigma.git
cd MySigma
```

### Create and activate a virtual environment

```bash
conda create -n MySigmaEnv python=3.12
conda activate MySigmaEnv
```

### Install dependencies

```bash
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1 --extra-index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

```bash
cd models/encoders/selective_scan && pip install --no-build-isolation . && cd ../../..
```

### Download weights

[Download link(Sigma-S-NYU)](https://drive.google.com/file/d/17afDv4BN69m66N3pfwTFnpBSXIUvlkwk/view?usp=drive_link)

## Running the code

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
or

```bash
python main.py
``` 
