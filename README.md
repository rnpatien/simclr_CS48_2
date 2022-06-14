# simclr_CS48_2

This project is a modelfied version of the TensorFlow model garden SIMCLR project. Refer to readme_SimClr.md for the project structure

Refer to https://www.tensorflow.org/guide/model_garden for documentation and https://github.com/tensorflow/models for code

This is a tensorflow model Garden application please load it in the official/vision/beta/projects/ directory.

## A new experiment has been added simclr_decode. 

It loads a trained simclr encoder (done in the exiting pretraining experiment) and trains a decoder to reconstruct the orginal image.


To train the decoder section of the auto-encode decoder, refer to the following command:

```
python3 -m official.vision.beta.projects.simclr.train \
  --mode=train_and_eval \
  --experiment=simclr_decode \
  --model_dir={MODEL_DIR} \
  --config_file={CONFIG_FILE}
```




