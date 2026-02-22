## 📊 CSV Reader & Auto-ML Pipeline

A powerful Django web application that allows users to seamlessly upload CSV datasets, automatically train machine learning models, and generate predictions—all through an intuitive user interface.

## 🚀 Overview

This project bridges the gap between raw data and actionable insights. By uploading a standard CSV file, users can automatically process their data, visualize feature correlations, train multiple machine learning algorithms, and deploy the best-performing model for immediate predictions without writing any code.

## ✨ Key Features

- **Automated Machine Learning:** Automatically detects the problem type (Classification vs. Regression) based on the selected target variable.
- **Smart Data Preprocessing:** Handles missing values, encodes categorical variables (label/one-hot encoding), and drops features with data leakage using Mutual Information.
- **Model Tuning & Selection:** Trains and evaluates multiple algorithms (Random Forest, Gradient Boosting, Logistic/Ridge Regression) and automatically saves the best performer.
- **Visual Analytics:** Generates dynamic distribution plots, correlation heatmaps, and feature importance charts using Seaborn and Matplotlib.
- **Interactive Predictions:** Dynamic web interface to input new feature values and receive real-time predictions from the trained models.
- **Modern UI:** Responsive and accessible frontend built with Tailwind CSS.

## 🛠️ Technology Stack

- **Backend:** Python, Django
- **Machine Learning & Data Processing:** Scikit-Learn, Pandas, NumPy, SciPy
- **Data Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, Tailwind CSS (`django-tailwind`)

## ⚙️ Local Setup & Installation

Follow these steps to get the project running on your local machine.

### Prerequisites
- Python 3.8+
- Node.js & npm (Required for Tailwind CSS)
- Git

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/csv_reader.git](https://github.com/yourusername/csv_reader.git)
cd csv_reader/csvreader
```
### 2. Set Up the Virtual Environment
```
Bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies
```
Bash
pip install -r requirements.txt
```
### 4. Install Tailwind CSS Dependencies
Navigate to the Tailwind static directory and install the required Node modules:
```
Bash
cd theme/static_src
npm install
cd ../..
```
### 5. Run Database Migrations
Initialize your local database:
```
Bash
python manage.py makemigrations
python manage.py migrate
```
### 6. Start the Development Servers
You will need to run the Django server and the Tailwind build process simultaneously. Open two terminal windows.
```
Terminal 1 (Compile Tailwind CSS):
Bash
python manage.py tailwind start


Terminal 2 (Run Django Server):
Bash
python manage.py runserver
```
### 7. Access the Application
Open your web browser and navigate to: http://127.0.0.1:8000

### 📖 Usage Guide
Sign Up / Log In: Create an account to access the dashboard.

Upload Data: Navigate to the upload section and submit a clean .csv file.

Select Target: Choose the column you want the AI to predict.

Train Model: Click "Start AI Training". The system will process the data, train the models, and redirect you to the results dashboard.

View Diagnostics: Explore the generated correlation heatmaps and feature importance charts.

Make Predictions: Enter new data points into the generated form to test your trained model.

### 🤝 Contributing
Contributions, issues, and feature requests are welcome!

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

### 📄 License
Distributed under the MIT License. See LICENSE for more information.