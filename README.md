# Steps to Create a Conda Environment with Python 3.10 (Named Qaddemly)

## Step 1: Create the environment

Run this command in your terminal:

```bash
conda create --name Qaddemly python=3.10
```

## Step 2: Activate the environment

```bash
conda activate Qaddemly

```

## step 3: install Libraries from requirments.txt

```bash
pip install -r requirements.txt --use-deprecated=legacy-resolver

```

or use

```bash
pip install --no-cache-dir --use-pep517 --upgrade --force-reinstall -r requirements.txt

```