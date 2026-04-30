## PIRecon Implementation

Official implementation of `Your Synthetic Image Is a Confession: Trojaning Image-to-Image Generative Models for Private Input Reconstruction`.

## Usage

1. Train a VQ model, or use any public pretrained VQ weights. We follow this repo to finetune the VQ model: https://github.com/CompVis/taming-transformers, and the pretrained weights we choose is vqgan_imagenet_f16_16384. 
2. Distill the VQ model into a VAE-encoder-style projection head by running:

   ```bash
   python PIRecon_projection_head_distill.py
   ```

3. Run stage 1 training:

   ```bash
   python PIRecon_stage1_training.py
   ```

   Load the distilled projection head in the script before training. Stage 1 uses curriculum training to reduce BER during embedding, and saves a checkpoint for stage 2.

4. Run stage 2 training:

   ```bash
   python PIRecon_stage2_training.py
   ```

   Load the checkpoint from stage 1 before training. Stage 2 improves the realism of the trojaned synthetic images.
