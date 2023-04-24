# Membership Inference Attacks fueled by Few-Shot Learning to deal with Data Integritye Attacks

This repo contains the code used to compose the experimental part of the paper.

To run this code with its dependenecies installed, we recommend to use the docker container [xehartnort/tfgpu](https://hub.docker.com/repository/docker/xehartnort/tfgpu/general).

Note that for the wikitext-103 experiments there are a few requirements, not provided in this repository due to their size:
- A path to "wikitext/wikitext.train" and "wikitext/wikitext.no_train", which contains the tokenized sentences of 1024 elements from wikitext-103 used to train GPT2. 
- A path to "wikitext/new_gpt2_model", which contains the checkpoint of the GPT-2 model trained for 20 epochs on "wikitext/wikitext.train".



