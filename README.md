# Credit Card Transactions Fraud Detection

<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="30%" src="https://github.com/ThuyTran102/1__Credit_Card_Transactions_Fraud_Detection/blob/main/images/credit_card.jpg" alt="creditcard"></img>
</p>

## Project Objective:
This project aims to develop a model that can predict or identify fraudulent credit card transactions based on the available features within a "Credit Card Transactions" dataset.

## Datasource:
The dataset used for this project is sourced from Kaggle.

You can find it here:
[Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)

This dataset is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration **1st Jan 2019 - 31st Dec 2020**. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

## Project Description:
This project comprises five Jupyter notebooks:

* 1-EDA_FeatureEngineering_Preprocessing.ipynb
* 2-ContinuingPreprocessing_FeatureSelection.ipynb
* 3-ModelSelection.ipynb
* 4-FineTuning_FinalPipeline.ipynb
* 5-FinalPipeline.ipynb

> **Note: Due to the large file size of some notebooks, particularly the first one, you might need to download them locally to view the code.**


## Key Steps Involved:
**1. Data Preprocessing**:
* Handling missing values, duplicates, etc.
* Scaling numerical variables and encoding categorical variables on the training set.
* Feature engineering to create new meaningful features.
* Exploratory data analysis using various visualization techniques.
* Using techniques like SMOTE to balance the classes in the training set.
* Feature selection to reduce dimensionality and improve model performance.

**2. Model Training**:
* Training multiple classifiers, including:
    - Logistic Regression
    - Support Vector Classifier (SVC)
    - Gaussian Naive Bayes (GaussianNB)
    - Random Forest Classifier
    - K-Nearest Neighbors Classifier
    - AdaBoost Classifier
    - Gradient Boosting Classifier
    - XGBoost
    - Bagging Classifier

**3. Model Evaluation and Selection**:
* Comparing models based on key metrics like precision, recall, F1 score, and ROC-AUC.
* Fine-tuning the selected model to optimize performance, especially with a focus on reducing false positives and false negatives.



## Project Outcomes:
Below are some key results from the model evaluation:

<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="60%" src="https://github.com/ThuyTran102/Credit-Card-Transactions_FRAUD-Detection/blob/main/images/Pipeline.png" alt="Outcome"></img>
</p>

<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="40%" src="https://github.com/ThuyTran102/Credit-Card-Transactions_FRAUD-Detection/blob/main/images/ROC_curve.png" alt="Outcome"></img>
</p>

<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="40%" src="https://github.com/ThuyTran102/Credit-Card-Transactions_FRAUD-Detection/blob/main/images/Confusion_matrix.png" alt="Outcome"></img>
</p>

<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="50%" src="https://github.com/ThuyTran102/Credit-Card-Transactions_FRAUD-Detection/blob/main/images/Classification_report.png" alt="Outcome"></img>
</p>


**Precision for Fraud Class (Class 1):** 0.97
> The precision is very high, indicating that most of the predicted fraud transactions are indeed fraudulent.

**Recall for Fraud Class (Class 1):** 0.90
> The recall is also strong, but there are still some fraudulent transactions (150 FN) that were not detected.
**F1 Score for Fraud Class (Class 1):** 0.94
> The F1 Score effectively balances precision and recall, which is critical for imbalanced datasets like this.

**Points to Consider:**
> The model leaves 150 fraudulent transactions undetected, which could have severe financial consequences. It may be beneficial to adjust the model’s decision thresholds or explore additional techniques to further reduce false negatives.


### Insights:
- Class imbalance is a significant challenge when detecting fraudulent transactions.
- Feature engineering and selection play a crucial role in boosting model performance.
- Precision is prioritized to reduce false positives, as detecting legitimate transactions incorrectly as fraud can negatively impact customer experience.

### Conclusion:
The current model strikes a good balance between precision and recall for fraud detection, with a focus on minimizing false positives. Future enhancements could involve refining the model to catch more fraudulent transactions without overly compromising precision.


### Future Directions:
If more time is available, I plan to explore integrating unsupervised learning techniques into this project. This could include methods like clustering to detect unusual transaction patterns that could further enhance fraud detection capabilities.

> Clustering helps group transactions with similar characteristics, enabling the creation of a new feature (i.e., cluster). This additional feature can provide valuable information about the data structure, helping improve fraud detection models. By introducing cluster labels as a feature, the model could better understand transaction patterns and potentially detect fraud more effectively.

These explorations in unsupervised learning would allow for a more nuanced approach to fraud detection, helping capture complex transaction patterns that traditional supervised learning methods might miss.
