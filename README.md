# Detection-of-Skin-disease-using-CNN
A deep learning model built to detect the skin disease


The steps involve in the project are:


1)	PREPROCESSING
The color image undergoes pre-processing step. In this step, morphological black hat transform is applied to remove epidermal hair.
The preprocessing code is included in preprocessing.py file


2)	BULDING CNN MODEL
The CNN model mainly performs two operations. They are feature extraction and classification. The main advantage of the CNN model is that the weights of the filter are automatically determined by the model itself during the training process.
The code for building the model is included in implementing_model.py file


3)	DEPLOYING MODEL
It is a framework in python. It is called micro framework as it doesn’t require tools and libraries. It is used to deploy the model on the web and render the HTML file.
The code using flask framework is included in deploy_model_flask.py file


4)	DEVELOPING A WEB TOOL
Once the model is built the weights of it can be saved in h5 file in which data is stored in Hierarchical Data Format (HDF). 
To serve the saved model and deploy it on the web, flask framework is used. Flask is based on Jinja2 template engine by which HTML file is rendered.
The code for front end using HTML and CSS is included in home1.html and result .html

