# Classification of Fraudulent Job Postings

Online Recruitment Fraud (ORF) is a way of employment scam where a person performs malpractice by posting fake job advertisements online, targeting job seekers who do not think about the legitimacy of the advertisement and reveal sensitive information to the fraudsters. There is a need to build models which can identify whether a job posting is fake or genuine to protect job seekers from their sensitive information falling into the wrong hands.

## Dataset

We used the [Real or Fake Job Postings dataset](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction). The dataset contains around 18,000 job postings out of which 800 are fraudulent. It contains meta-data and textual information about the jobs. It contains information such as ‘company_profile’, ‘description’, ‘benefits’ as textual information and ‘has_company_logo’, ‘has_questions’ as binary value data. The last column, named ‘fraudulent’, contains 0s and 1s – 0 means the posting is a real job posting and 1 means the job posting is a fraudulent one. The original and pre-processed datasets are available [here](https://drive.google.com/file/d/1zW2YIDvTU6SCe6J2qekiBB38xUEtQHqB/view?usp=sharing)

## Getting Started

Make sure the following python libraries/ modules are available on your system
* sklearn
* matplotlib
* pandas
* numpy

## General Approach

To compare classification models trained on preprocessed dataset that contains combined data from all other columns in the form of text. Preprocessing is done to ensure even the presence or absence of data and values of most of the columns are accounted for.

## EDA and Visualization

Exploratory data analysis brought us a few useful insights into our dataset. The code is outlined in this [notebook](https://github.com/NehaKohad/DataAnalytics-We-re_Skewed/blob/master/EDA_and_Visualization.ipynb).Our dataset has an imbalance with very few job postings of fake job postings. We dealt with duplicates and missing data as outlined in <put link>. Assessing every column involved replacing the NaN values with "missing" as we wanted to retain the information about presence and absence of data as well.

## Data Preprocessing

Load the dataset using pandas. As per conclusions from the graphs plotted during EDA, we dropped 2 columns - job_id and salary_range due to the high number of missing values. The categorical variables containing textual data were left untouched. However, those with 1 and 0 were replaced to ensure even the presence or absence of data and values of most of the columns are accounted for. For example, 'telecommuting' contains 1 to indicate telecommuting was required and 0 to indicate it was not a requirement. They were replaced with values as follows :
Value  |Replacement
| :--- | :---
1  | telecommuting
0  | no telecommuting

After this, columns were all combined into one column - 'text' where regex was used for further cleaning. We removed punctuations, unwanted spaces, special characters and so on. The resulting text column was now lemmatized. The script is available [here](https://github.com/NehaKohad/DataAnalytics-We-re_Skewed/blob/master/Data%20Preprocessing.ipynb).

## Model Building
### Splitting Data into Train and Test Sets

Since the dataset is imbalanced, we chose to do cross validation using stratified k-fold. It splits the dataset into k batches while preserving relative percentages of each class of target variable. The training split is vectorized using TF-IDF For every fold, the data is split into training and testing dataframes. The model building code is available [here](https://github.com/NehaKohad/DataAnalytics-We-re_Skewed/blob/master/Model%20Building.ipynb).

### Model

We compared performance of the following classfiers using accuracy and specificity.
* K Nearest Neighbours (KNN)
* Random Forest (RF)
* Multinomial Naive Bayes (MNB)
* Logistic Regression (LR)
Going through each fold iteratively, train and test splits were generated. Every classifier was trained and tested on the current split. Accuracy and Specificty was recorded per classifer per fold for comparison. 

## Evaluation

Accuracy and specificty were used as performance metrics to make comparisons. However, specificty was given more importance given our dataset and it's structure. The target column fraudulent indicates a real job posting with 1 and a fake one with 0. True negatives in our scenario indicate the number of fake job postings that were classified as fraudulent. Specificity, defined as TN / (TN + FP) gives us a better indication of teh performance of classifer as compared to accuracy.


## Results
Agggregate of the accuracy and specificty after k-folds is calculated. Following results were obtained after stratified 5-fold cross validation :
Classifier | Accuracy | Specificity
| :--- | ---: | :---:
K Nearest Neighbours  | 97.87  | 97.5
Logistic Regression  | 96.74  | 99.6
Random Forest  | 98.23  |  97.4
Multinomial Naive Bayes  | 98.62 |  94.55


## Instructions for execution

- The Data Prepocessing.ipynb file contains the data preprocessing and cleaning code. The csv file used to run the code is 'fake_job_postings.cvs' .
  Running this file generates a new csv file 'preprocessed.csv'
- The Model_Building.ipynb is the next file to be executed. It uses the 'preprocessed.csv' file that was generated earlier. Running the code displays the results mentioned above.
- EDA_Visualization.ipynb contains the code that generates the graphs found in the folder 'Graphs'.
