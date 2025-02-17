# Liver_Disease_Prediction_Using_Federated_Learning


A Python-based Federated Learning (FL) framework that enables distributed model training across multiple clients while preserving data privacy. This project simulates a federated learning setup with multiple clients and a central server coordinating the training process.

## Features

- Implements a federated learning architecture with a central server and multiple clients.
- Supports data partitioning for simulating decentralized data sources.
- Provides utilities for organizing and managing test data.
- Designed for easy experimentation and extension.

## Setup & Usage

1. Install required dependencies:
   ```sh
   pip install torch
   pip install pandas
   pip install flask
   pip install tensorflow numpy pickle-mixin streamlit pillow matplotlib
   

   ```
2. Start the server:
   ```sh
   python _server.py
   ```
3. Run client scripts:
   ```sh
   python client_1.py 
   python client_2.py 
   python client_3.py 
   python client_4.py 
   ```
4. Run Frontend:
   ```sh
   streamlit run app.py
   ```
      



