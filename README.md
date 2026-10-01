# Multi-Vector Threat Detection System

## Overview

The Multi-Vector Threat Detection System is an AI/ML-based cybersecurity application designed to detect different types of digital threats such as phishing URLs, Business Email Compromise (BEC), and image-based steganography.

The system combines Machine Learning, Natural Language Processing, Deep Learning, and rule-based analysis to provide a multi-layered approach to threat detection.

## Objectives

- Detect malicious and phishing URLs.
- Identify suspicious or potentially compromised emails.
- Detect hidden information in images using steganography analysis.
- Provide users with a simple web interface for threat analysis.
- Combine multiple detection techniques into a single security application.

##  Key Features

### 1. Phishing URL Detection
Analyzes URL characteristics such as:
- URL length
- Domain information
- IP address usage
- Special characters
- Suspicious patterns
- Domain similarity

**Algorithm:** Random Forest

### 2. BEC / Email Detection
Analyzes email content to identify suspicious or potentially malicious emails.

**Techniques:**
- Text preprocessing
- TF-IDF feature extraction
- Multinomial Naive Bayes classification

### 3. Image Steganography Detection
Analyzes images to identify whether hidden information may be present.

**Approach:**
- Image preprocessing
- Feature extraction
- Deep Learning
- ResNet18 CNN

### 4. 🛡️ Multi-Vector Analysis
The system combines multiple threat detection vectors:

**URL → Email → Image**

This provides a broader approach to cybersecurity threat detection.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- Bootstrap
- Scikit-learn
- Pandas
- NumPy
- TensorFlow / PyTorch
- ResNet18
- TF-IDF
- Random Forest
- Naive Bayes
- Machine Learning
- Deep Learning

## Project Structure

```text
Multi-Vector-Threat-Detection-System/
│
├── datasets/
├── models/
├── modules/
├── static/
├── templates/
│
├── app.py
├── check_all_models.py
├── database.db
└── README.md
