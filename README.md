# fetch-take-home-assignment

## Name: Atharv Patwardhan

The files are named according to the task written inside them.

task1.py contains task 1, task2.py contains **BOTH** task 2 and the task 4 training loop, and task3.txt contains task 3.

## Overview of Multi-Task Sentence Transformer

This project implements a custom Sentence Transformer extended with Multi-Task Learning for:

- Task A: Sentence Topic Classification (sports, tech, politics)
- Task B: Sentiment Analysis (positive, neutral, negative)

---

## Project Structure

- `tasks/`: Individual task tests and explanations
- `Dockerfile`: Reproducible container setup

---

## Getting Started

### Install (Locally)

```bash
pip install -r requirements.txt
python run_training.py
```

### Or run via docker

```bash
docker build -t fetch-assesment .
docker run --rm -it fetch-assesment
```
