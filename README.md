#🧹 CleanML – Smart Data Cleaning Application

CleanML is a Python-based web application built with Streamlit that helps users clean and prepare datasets for Machine Learning workflows.

It provides automatic and customized data-cleaning options, data-quality analysis, visualizations, and downloadable cleaning reports — all through a simple and interactive interface.

🚀 Features

Dataset Upload: Upload CSV, XLSX, and XLS files.

Dataset Preview: View uploaded data before cleaning.

Data Quality Analysis: Identify missing values, duplicate rows, numerical columns, and categorical columns.

Machine Learning Algorithm Selection: Select from Linear Regression, Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and Naive Bayes.

Missing Value Treatment: Handle missing numerical and categorical values using different strategies.

Duplicate Removal: Identify and remove duplicate records.

Outlier Treatment: Detect and handle outliers using IQR-based methods.

Categorical Encoding: Apply Label Encoding or One-Hot Encoding.

Feature Scaling: Apply StandardScaler or MinMaxScaler where appropriate.

Customized Cleaning: Choose your own cleaning methods and settings.

Before-and-After Comparison: Compare dataset quality before and after cleaning.

Download Results: Export cleaned datasets in CSV and Excel formats.

PDF Cleaning Report: Download a report containing cleaning results and operations performed.

Cleaning History: Review and download the cleaning operations history.

🛠️ Technologies Used

Technology

Purpose

Python

Programming language

Streamlit

Web application interface

Pandas

Data manipulation

NumPy

Numerical operations

Scikit-learn

Feature preprocessing and scaling

Plotly

Interactive visualizations

OpenPyXL

Excel file support

xlrd

XLS file support

ReportLab

PDF report generation

📂 Project Structure

CleanML/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/                 # Local virtual environment (not uploaded)

⚙️ Installation and Setup

1. Clone the repository

git clone https://github.com/priyanka-sapkal101/CleanML-Smart-Data-Cleaning.git

2. Open the project folder

cd CleanML-Smart-Data-Cleaning

3. Create a virtual environment

python -m venv venv

4. Activate the virtual environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1

5. Install dependencies

pip install -r requirements.txt

6. Run the application

streamlit run app.py

The application will open in your web browser.

🔄 Application Workflow

Upload a dataset.

Preview the uploaded data.

Analyze dataset quality.

Select a Machine Learning algorithm.

Choose the target column.

Apply automatic cleaning or customize cleaning settings.

Review cleaned data and compare results.

Download the cleaned dataset and cleaning report.

📊 Output

CleanML provides:

Cleaned CSV dataset

Cleaned Excel dataset

PDF cleaning report

Cleaning history in CSV format

Before-and-after dataset comparison

🎯 Project Objective

The main objective of CleanML is to simplify the data preprocessing stage of Machine Learning by providing an interactive tool that helps users identify and handle common data-quality issues.

It aims to reduce repetitive manual data-cleaning work and make datasets easier to prepare for further Machine Learning tasks.

⚠️ Important Note

CleanML focuses on data cleaning and preprocessing. It does not train or evaluate the selected Machine Learning algorithm.

For reliable model evaluation, preprocessing transformations should be fitted on the training dataset only to avoid data leakage.

👩‍💻 Author

Priyanka Sapkal

GitHub: priyanka-sapkal101

⭐ If you find this project useful, feel free to star the repository!
