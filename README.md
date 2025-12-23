# **Biological Age Estimation Framework**

This repository contains the official implementation of the code for the paper: **"Unsupervised Large Language Model for Aging Evaluation based on Sparse Rural Health Data"**.

This framework utilizes Large Language Models (LLMs) via Ollama to estimate biological age and system-specific aging patterns based on multidimensional clinical health records.

## **📋 Table of Contents**

* Prerequisites 
* Installation
* Data Format
* Usage 
* License

## **🛠 Prerequisites**

### **1\. System Requirements**

* Python 3.8+  
* [Ollama](https://ollama.com/) installed and running locally (or accessible via network).

### **2\. LLM Setup**

This code depends on Ollama. Please ensure you have pulled the model you intend to use (e.g., Llama 3, Mistral, MedLlama).

\# Example: Pulling Llama 3.1 70b  
echo "ollama pull llama3.1:70b"

## **📦 Installation**

1. Clone this repository:  
   git clone \[https://github.com/CaCO3-666/Unsupervised-Large-Language-Model-for-Aging-Evaluation-based-on-Sparse-Rural-Health-Data.git\](https://github.com/CaCO3-666/Unsupervised-Large-Language-Model-for-Aging-Evaluation-based-on-Sparse-Rural-Health-Data.git)  
   cd Unsupervised-Large-Language-Model-for-Aging-Evaluation-based-on-Sparse-Rural-Health-Data

2. Install Python dependencies:  
   pip install \-r requirements.txt

## **📂 Data Format**

Due to patient privacy regulations, the original dataset used in the paper cannot be shared. The input CSV must contain at least the following columns:

* person\_id: Unique identifier for the individual (Integer).  
* health\_record: A text string containing the clinical data.

## **🚀 Usage**

### **1\. Start Ollama Server**

Ensure your Ollama server is running. By default, the code expects it at http://localhost:11434.

### **2\. Run the Estimation**

You can run the script using the command line. You can configure the parameters via environment variables or modify the script defaults.

**Basic usage with dummy data:**

\# Set the model name you want to use  
export OLLAMA\_MODEL="llama3.1:70b"  
export OLLAMA\_HOST="http://localhost:11434"

\# Run the script  
python main\_code.py \--input dummy\_data.csv \--output results.csv

**Arguments:**

* \--input: Path to the input CSV file.  
* \--output: Path to save the result CSV file.  
* \--limit: (Optional) Limit the number of records to process for testing.  
* \--checkpoint-every: (Optional) Save results every N records (default: 1).

## **📄 License**

This project is licensed under the MIT License \- see the LICENSE file for details.
