# Lightweight and Dynamic Deblurring for IoT enabled Smart Cameras

## Abstract
[IEEE Internet of Things Journal](https://ieeexplore.ieee.org/document/9776515)

Tensorflow implementaion of **"Lightweight and Dynamic Deblurring for IoT-enabled Smart Cameras"** by Ju-Wei, Que. 

This repo including speed-oriented & quality-oriented image deblurring models for usage. <br>

According to loss functions composition, **Direct-mapping** type & **GAN-based** type are introduced.
  - Speed-oriented model
    - Direct-mapping type
    - GAN-based type
 
  - Qaulity-oriented model
    - Direct-mapping type
    - GAN-based type  

  - Refer the paper for details.

## Model Architecture Overview

- Speed-oriented model architecture design:
![image](https://user-images.githubusercontent.com/35868815/174267732-f6a64672-640e-45ec-8261-7e377106269e.png)

- Qaulity-oriented model architecture design:
![image](https://user-images.githubusercontent.com/35868815/174268689-ca31fa8c-ed76-4d47-8835-9c40ab5ec9b6.png)

## How to use

- For training
  - Entry point: ./pipeline/train.py
  - Config: ./train_cfg/cfg.txt
    - including four model sections related info
  - CMD: `python train.py --cfg_path ../train_cfg/cfg.txt --sectioon SpeedOriented_DirectMapping`
- For testing
  - Entry point: ./pipeline/test.py
  - Config: ./test_cfg/cfg.txt
    - Given testing data path, model ckpt path, training model section for testing
  - CMD: `python test.py --cfg_path ../test_cfg/cfg.txt --section Testing`

## Demo

## Comparison

## Acknowledgments
