# AGENTS.md

## Repository Focus
This repository contains both:
1. the current forecasting project for probabilistic EV charging demand forecasting
2. some legacy code from earlier allocation / MCS-related work

For the current submission, focus on the forecasting pipeline only.

## Read First
When starting work, read these files first:
1. `project.md`
2. `train.py`
3. `train.sh`
4. files under `utils/model_training/`
5. forecasting-related plotting scripts

## Current Priority
The immediate priority is the conference submission in about 10 days.

Focus on:
- understanding the forecasting pipeline
- supporting lightweight evaluation scripts
- computing new metrics from existing predictions
- generating high-quality plots for the paper
- suggesting realistic extension ideas with low implementation cost

## Things To Ignore For Now
- legacy allocation / MCS modules unless explicitly requested
- large-scale refactoring
- redesigning the backbone architecture
- expensive retraining that is unlikely to finish before the deadline

## Code Change Rules
- prefer minimal and targeted changes
- explain the relevant code path before changing files
- avoid modifying unrelated modules
- preserve reproducibility whenever possible
- save outputs in a clear and paper-oriented way

## Research Support Rules
When proposing ideas:
- stay close to the current implemented method
- prioritize feasible extensions
- consider deadline, code cost, and paper value together