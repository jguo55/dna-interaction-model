# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is intended to predict molecule and DNA interactions. It uses the .csv files in the data folder for the interactions, and the sequences for each gene are stored in the genes.hdf5 folder.

## Development Commands

- **Run the main script**: `python main.py`
- **Check Python syntax**: `python -m py_compile main.py`
- **Format code**: `python -m black main.py` (if black is installed)
- **Lint code**: `python -m flake8 main.py` (if flake8 is installed)

## Architecture

Data is in the data folder. Main.py trains the model. The dna sequences are loaded each time during training due to their large sizes.