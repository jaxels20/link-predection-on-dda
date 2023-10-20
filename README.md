# Graph-based Drug-Repurposing with Negative Edge Sampling Techniques

## Overview
This repository contains our study on using graph machine learning for drug-repurposing tasks. The main focus is exploring the impact of various negative edge sampling techniques on the performance of the models.

## Dataset
- Derived from Medication Reference Terminology.
- Consists of 4,282 drugs, 1,308 diseases, and 14,686 associations.

## Techniques Explored
1. Uniform negative edge sampling before batching.
2. Uniform negative edge sampling after batching.
3. Generator model from a generative adversarial network before batching.

## Key Result
- Best model achieved a mean AUC of 0.880 (n = 20) using uniform negative edge sampling after batching.



