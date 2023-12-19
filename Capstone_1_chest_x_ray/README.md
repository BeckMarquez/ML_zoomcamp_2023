# AI Model for Pneumonia Detection Using Chest X-Ray Images

## 1. Problem Description

According to the World Health Organization (WHO), pneumonia kills about 2 million children under 5 years old every year and is consistently estimated as the single leading cause of childhood mortality (Rudan et al., 2008), killing more children than HIV/AIDS, malaria, and measles combined (Adegbola, 2012). The WHO reports that nearly all cases (95%) of new-onset childhood clinical pneumonia occur in developing countries, particularly in Southeast Asia and Africa. Bacterial and viral pathogens are the two leading causes of pneumonia (Mcluckie, 2009) but require very different forms of management.

## 2. Possible uses of solutions

Bacterial pneumonia requires urgent referral for immediate antibiotic treatment, while viral pneumonia is treated with supportive care. Therefore, accurate and timely diagnosis is imperative. One key element of diagnosis is radiographic data, since chest X-rays are routinely obtained as standard of care and can help differentiate between different types of pneumonia. However, rapid radiologic interpretation of images is not always available, particularly in the low-resource settings where childhood pneumonia has the highest incidence and highest rates of mortality. These why AI model may help to determine pneumonia as urgent as possible.

## 3. Dataset Description

The dataset we use is from [Kaggle.com](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The dataset is quite large and not included directly to project repository. It consists of:
- total training normal images: 1342
- total training pneumonia images: 3876
- total validation normal images: 8
- total validation pneumonia images: 8
- total test normal images: 234
- total test pneumonia images: 390

## 4. Project files

1. Dockerfile - file to build Docker image for running the service
1. img_for_test - folder with 4 images that have been used for testing
1. chest_x_ray.tflite - TF Lite model. There were other models, but I can\`t pull them to Github because of size. But you can create them if you execute `notebook.ipynb`
1. README.md - this file
1. Creation_TFlite_model.ipynb - from .h5 to TF Lite model conversion and deleting tensorflow dependencies
1. notebook.ipynb - jupyter notebook containing data preparation, EDA, model selection with model fine-tuning
1. predict.py - python script with model loading and serving it via a web service (Flask)
1. requirements.txt - file with all venv containers used in project
1. testing_TFlite_model_deploy_google_run.ipynb - notebook for testing model deployed on Google Run
1. testing_TFlite_model_flask.ipynb - notebook for testing model using Flask
1. testing_TFlite_model_locally.py - script for simple local model testing
1. train.py - python script containing the final model training

## 5. How to run application using Google Run

1.App is deployed on Google Run [link](https://chest-x-ray-txo26qljya-ew.a.run.app/predict), but it won't open through browser. 
You must run `testing_TFlite_model_deploy_google_run.ipynb` from your python IDE.
> [!NOTE]
>
> first time launching takes time, if you got an error, rerun notebook after 10 minutes

## 6. How to build a contained application and how to run it

Application is containerized, you must have Docker desktop installed on your PC.

1. Download files from "Midterm_project" branch and place them to separate folder
2. Run terminal on this folder
3. Build Docker image using command (don`t forget dot at the end):

```
docker build -t chest_x_ray_tflite_model .
```
or 
you may pull image from Docker hub using command:
```
docker pull beckmarquez/chest_x_ray
```
4. Run created image using command (don`t forget dot at the end):

```
docker run -i --rm -p 9696:9696 chest_x_ray_tflite_model:latest
```

5. You should see the following:

```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
INFO:waitress:Serving on http://0.0.0.0:9696
```

6. Now you may run `testing_TFlite_model_flask.ipynb` from your python IDE
7. When you are done press "Ctrl+C" in your terminal to stop process

## 7. How to build environment used in the project

In Unix ow WSL:
1. `cd` to your desirable dev folder
2. Run commands:
`pip install virtualenv` > if you don't already have virtualenv installed
`virtualenv venv` > to create your new environment (called 'venv' here)
`source venv/bin/activate` > to enter the virtual environment
`pip install -r requirements.txt` > to install the requirements provided in `requirements.txt` in the current environment
`deactivate` > to exit from venv

If you using Conda, problems may appear, because currently (December 2023) Conda uses Tensorflow 2.10 and in these project I have been using Tensorflow 2.15