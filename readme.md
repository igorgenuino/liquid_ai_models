 ✅ Prerequisites
Before you begin, please make sure you have the following installed on your system:

Python: You'll need Python 3.8 or newer. You can download it from python.org.

PIP: This is Python's package installer and usually comes with Python.

⚙️ Setup Instructions
Follow these steps to set up the project environment.

1. Arrange Files and Folders
The Python scripts need to be in the same directory as your model files. Please organize your project folder exactly like this:

your_main_project_folder/
│
├── 📂 LFM2-VL-450M/
│   └── (model files for LFM2-VL...)
│
├── 📂 SmolVLM2-500M-Video-Instruct/
│   └── (model files for SmolVLM2...)
│
├── 📜 running_model_original_lfm2_vl.py
├── 📜 running_model_original_smolVLM2.py
├── 🖼️ logo.png
└── 🖼️ icon.png

2. Install Required Python Libraries
You can install all the necessary libraries automatically using a requirements.txt file.

Install the libraries:
Open a terminal or command prompt, navigate to your_main_project_folder, and run the following command:

Bash

pip install -r requirements.txt

▶️ How to Run the Applications
Once everything is set up, you can run each application.

Open your terminal or command prompt and navigate to your main project folder.

To run the LFM2-VL model, execute this command:

Bash

python running_model_original_lfm2_vl.py
To run the SmolVLM2 model, execute this command:

Bash

python running_model_original_smolVLM2.py
After running either command, the script will start loading the model (which may take a minute). Once it's ready, it will print a local URL in the terminal, like http://127.0.0.1:7860. You can open this URL in your web browser to use the application.

Extra
For windows there are .bat files to install_requirements and run the models.

Info about files
-python running_models_... will run the models and open the browser
-fine_tunning_.. will run the fine tune
-run