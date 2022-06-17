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
 <br>
  - Refer the paper for details.

## Model Architecture

## How to use

- For training
  - Entry point: ./pipeline/train.py
  - Config: ./train_cfg/cfg.txt
    - including four model sections related info
- For testing
  - Entry point: ./pipeline/test.py
  - Config: ./test_cfg/cfg.txt
    - including model ckpt path, model section for testing

## Demo

## Comparison

## Acknowledgments
