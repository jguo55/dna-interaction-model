# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is intended to predict molecule and DNA interactions. It uses the .csv files in the data folder for the interactions, and the sequences for each gene are stored in the genes.hdf5 folder.

## Development Commands

- **Run the main script**: `python train.py`
- **Check Python syntax**: `python -m py_compile train.py`
- **Format code**: `python -m black train.py` (if black is installed)
- **Lint code**: `python -m flake8 train.py` (if flake8 is installed)

## Architecture

Data is in the data folder, ignore the raw_data folder (it's where the dataset is created from). Train.py trains the model. This project is intended to be built into a docker container, then trained using kubernetes and validate-job.yaml.