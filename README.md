# AI_Design_Thermal_Battery

This is the code for paper "
Elastocaloric Thermal Battery:  AI-Designed Alloys for Efficient Waste Heat Recycling". 

Alloy_generation.py: main function: candidate alloy compositions and processing parameters are proposed by a CVAE generator, constrained by a hand-drawn heat flow curve, and subsequently filtered using a XGBR predictor.

model_CVAE: fold that stores the trained generative model.

result: fold that stores generated compoitions and process parameters.

![390db9e5699d8c984f063eed8c7e41f](https://github.com/user-attachments/assets/7e7d4f76-6ef2-4237-a57f-64077acec8cc)
