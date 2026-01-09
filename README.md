# Resume-Screening
🌐 Resume Screening Web Application (NLP + Gemini AI)

This repository contains the main website project for an AI-powered Resume Screening System, developed using Streamlit, Machine Learning, and Google Gemini AI.

The application allows users to upload resumes, compare them with a job description, classify resumes, calculate match scores, and generate short professional summaries.

📌 Note:
Model training is NOT performed in this project.
Training and experimentation were done separately in a Jupyter Notebook (.ipynb), which is uploaded as a supporting project and report.

⸻

🚀 Website Features
	•	Upload multiple PDF resumes
	•	Paste a job description
	•	Predict resume job category
	•	Calculate resume–JD match score
	•	Generate AI-powered 2-line summaries
	•	Rank resumes automatically
	•	Download results as CSV / PDF
	•	Secure Gemini API key usage

⸻

🧠 Technology Stack
	•	Frontend: Streamlit
	•	NLP: TF-IDF Vectorizer
	•	ML Model: Logistic Regression (One-vs-Rest)
	•	AI Summary: Google Gemini
	•	PDF Parsing: PyPDF2
	•	Similarity Metric: Cosine Similarity

⸻

📁 Project Structure (Website)
.
├── ResumeScreeningStreamlitApp.py   # Main website application
├── role_classifier_ovr_lr.joblib       # Pre-trained ML model
├── tfidf_vectorizer.joblib             # Pre-trained vectorizer
├── founder/
│   └── blackHoodieAryanAgarwal.png
├── LICENSE
└── README.md

🖥️ How to Run the Website on Your System

✅ Step 1: Install Python
Ensure Python 3.9+ is installed.
python --version
✅ Step 2: Clone the Repository
git clone https://github.com/your-username/resume-screening-website.git
cd resume-screening-website
✅ Step 3: Configure Gemini API Key
Streamlit Secrets (Recommended)
Create:
.streamlit/secrets.toml
Add:
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
✅ Step 4: Run the Website
streamlit run ResumeScreeningStreamlitApp.py

🧩 How the Website Works
	1.	User uploads resume PDFs
	2.	Resume text is extracted
	3.	Job description is processed
	4.	TF-IDF vectors are compared
	5.	Match score is calculated
	6.	Resume category is predicted
	7.	Gemini AI generates summary
	8.	Results are ranked and displayed

⸻

📘 About Model Training (Supporting Work)

The machine learning models used in this website were trained outside this project using a Jupyter Notebook (.ipynb).

This repository may also include:
	•	📓 A training notebook
	•	📄 A report explaining model training

These files are for documentation and learning purposes only.

Training Project Includes:
	•	Data preprocessing
	•	TF-IDF feature engineering
	•	Model experimentation
	•	Logistic Regression training
	•	Model evaluation
	•	Exporting .joblib files

🚫 The website does not retrain models.

⸻

📄 About the Training Report

The training report explains:
	•	Why Logistic Regression was chosen
	•	How features were engineered
	•	Model evaluation approach
	•	Final model selection reasoning

This ensures transparency and reproducibility.

⸻

🔐 Security Notes
	•	API keys are stored securely
	•	.streamlit/secrets.toml must be ignored in Git
	•	No sensitive data is committed

⸻

👨‍💻 Developer

Aryan Agarwal
🔗 https://www.linkedin.com/in/aryanagarwal-dev/

⸻

📜 License

This project is intended for academic and educational use.
Please review Google Gemini API terms for commercial usage.

⸻

⭐ Summary
	•	🌐 Main Project: Streamlit Resume Screening Website
	•	📓 Supporting Project: Model training notebook
	•	📄 Supporting Report: Training documentation
	•	🚀 Production-ready inference app
