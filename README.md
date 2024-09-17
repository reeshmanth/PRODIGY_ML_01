# PRODIGY_ML_01
This project aims to predict house prices based on their square footage, number of bedrooms, and number of bathrooms. A linear regression model is implemented using Python. The model is trained on a dataset and evaluated using metrics like mean squared error to assess its accuracy.

House Price Prediction

This project implements a linear regression model to predict house prices based on features such as square footage, number of bedrooms, and number of bathrooms. The model is built using Python and popular data science libraries.

Table of Contents
-Installation
-How to Run
-Data Description
-Model Evaluation
-License

Installation
To run this project, you will need the following Python libraries:
-pandas
-numpy
-matplotlib
-seaborn
-scikit-learn

You can install them using pip:
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run
Clone the repository:
git clone https://github.com/reeshmanth/house-price-prediction.git
Navigate to the project directory:
cd House Price Prediction
Run the main script using Python:
python house_price_prediction.py

Data Description
The project uses two datasets:
train.csv: The training dataset containing features such as GrLivArea, BedroomAbvGr, FullBath, and the target variable SalePrice.
test(2).csv: The test dataset for making predictions. It contains similar features but does not include the target variable.

Model Evaluation
The model's performance is evaluated using Mean Squared Error (MSE) on a validation set. The lower the MSE, the better the model's predictions.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
