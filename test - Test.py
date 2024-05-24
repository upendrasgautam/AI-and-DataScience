
# Importing libraries
from PyPDF2 import PdfReader
import re
import PyPDF2
import os
from os.path import isfile
from os.path import join
import shutil
import tensorflow as tf
import cv2
import numpy as np
import pytesseract
import pandas as pd
import sys
import subprocess
import re
from pytesseract import Output
import json
import logging
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

def process_data_and_return_json(pdf_path, image_folder, document_folder, model_path, reject_model_path, detection_folder):
    
    # Set the logging verbosity level to suppress info and below
  
    tf.get_logger().setLevel('INFO')
    logging.getLogger("tensorflow").setLevel(logging.INFO)
    
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    model = tf.keras.models.load_model(model_path)
    train_labels = ['Account Closed', 'Already Paid', 'Customer is deceased', 'Incorrect check amount',
                        'Insufficient Funds', 'Invalid Account number', 'Invalid address', 'Invalid Loan number',
                        'Invalid payee', 'Invalid payment method', 'Need more information', 'No Balance due', 'No reason',
                        'Overpaid', 'Paid in full', 'Payee no longer manages this account',
                        'Payee no longer manages this loan service', 'Policy is cancelled', 'Received in error',
                        'Refused by payee', 'Suspected Fraud', 'Unable to locate account']

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    max_words = 10000
    max_len = 100
    tokenizer = Tokenizer(num_words = max_words)


    def extract_images_from_pdfs(pdf_path, image_folder):
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.makedirs(image_folder)
    
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        reader = PdfReader(pdf_path)
    
        for page_number, page in enumerate(reader.pages):
            for image_number, image in enumerate(page.images):
                image_data = image.data
                image_filename = f"{filename}_page{str(page_number + 1).zfill(3)}.png"
                output_path = os.path.join(image_folder, image_filename)
                with open(output_path, "wb") as fp:
                    fp.write(image_data)
                    

     # Function for Image Classification in checks & letters
    def classify_image(image_path):   
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (224, 224))
        normalized_image = resized_image / 255.0
        external_image = np.expand_dims(normalized_image, axis=0)
        predictions_cr = model.predict(external_image)
        return np.argmax(predictions_cr)

    # Create dataframe where checks in images folder & letters in document folder & backfilling the reason_image_path
    def process_classification_results(image_folder, document_folder):
        image_files = os.listdir(image_folder)
        data = []
        current_check = None
        current_reason = None
    
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            classification_result = classify_image(image_path)
            # print(classification_result)
    
            if classification_result == 0:
                if current_check is not None:
                    data.append({'check_image_path': current_check, 'reason_image_path': current_reason,
                                 'check_name': os.path.basename(current_check)})
    
                current_check = image_path
                current_reason = None
            elif classification_result == 1:
                # Move image to document_folder
                new_path = os.path.join(document_folder, os.path.basename(image_path))
                shutil.move(image_path, new_path)
    
                if current_reason is None:
                    current_reason = new_path
                else:
                    current_reason += ', ' + new_path
    
        if current_check is not None:
            data.append({'check_image_path': current_check, 'reason_image_path': current_reason,
                         'check_name': os.path.basename(current_check)})
    
        df = pd.DataFrame(data)
        
        if len(df) == 1:
            # Handle single-row DataFrame separately
            df['reason_image_path'] = df['reason_image_path'].fillna('')  # or any other default value
        elif not df['reason_image_path'].empty:
            # If it's not empty, fill missing values using backward filling
            df['reason_image_path'] = df['reason_image_path'].fillna(method='bfill')
    
        return df

    def run_object_detection_script(current_image_path):
        
        script_args = [
            "detect.py",
            "--weights",
            "yolov7_custom.pt",
            "--conf",
            "0.4",
            "--img-size",
            "640",
            "--source",
            "",
            "--no-trace",
            "--save-txt",
        ]

        script_args[script_args.index("--source") + 1] = current_image_path

        command = ["python"] + script_args

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"Script executed successfully for {current_image_path}:\n{stdout}")
        else:
            print(f"Error executing the script for {current_image_path}:\n{stderr}")

    
    # Run detection on each row in DataFrame  
    def run_detection_on_dataframe(df):
        if "check_image_path" not in df.columns:
            raise ValueError("DataFrame must contain a column named 'check'.")
        results = []
        for _, row in df.iterrows():
            image_path = row["check_image_path"]
            extracted_details = run_object_detection_script(image_path)
            if extracted_details is not None:
                results.append(extracted_details)
        return results.append(extracted_details)
   
    # Tesseract extraction on Image basis the Coordinates
    def crop_image(x1, y1, x2, y2, image):
        crop_image = image[y1:y2, x1:x2]
        
        # grayscale
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        
        # threshold the image using Otsu's thresholding method
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        # Noise removal
        denoised_image = cv2.medianBlur(thresholded, 1)
        
        results = pytesseract.image_to_data(denoised_image, lang='eng', output_type=Output.DICT)
         
        if results is not None:
            text = ' '.join(results['text']).strip()
        else:
            text = ""
        return text

    # Cropping Images & Creating dictionary for check Info
    def process_check(image, file_path):
        image = cv2.imread(image)
        list_of_lists = []

        with open(file_path, 'r') as file:
            for line in file.readlines():
                numbers = list(map(int, line.strip().split()))
                list_of_lists.append(numbers)

        # Define the mapping of keys
        key_mapping = {
            0: 'Payee Account Number ',
            1: 'Subscriber',
            2: 'Payee',
            3: 'FI',
            4: 'CheckNo',
            5: 'Date',
            6: 'Amount',
            7: 'Validity'
        }
        datadict = {}
        for val in list_of_lists:
            data = crop_image(*val[1:], image)
            datadict[key_mapping[val[0]]] = data.strip()
        return datadict

    
        
        
    # def combine_images(image_paths, output_path):
    #     images = [cv2.imread(image_path) for image_path in image_paths]
        
    #     # Ensure all images have the same number of channels
    #     max_channels = max(image.shape[2] if len(image.shape) == 3 else 1 for image in images)
    #     images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 and max_channels == 1 else image for image in images]
        
    #     # Ensure all images have the same dimensions
    #     max_height = max(image.shape[0] for image in images)
    #     max_width = max(image.shape[1] for image in images)
    #     images = [cv2.resize(image, (max_width, max_height)) for image in images]
        
    #     combined_image = cv2.hconcat(images)
    #     cv2.imwrite(output_path, combined_image)

    # checking if detection_folder already exists
    try:
        shutil.rmtree(detection_folder)
        os.makedirs(detection_folder)
        # print(f"Folder '{detection_folder}' cleared successfully.")
    
    except Exception as e:
        pass
        # print(f"An error occurred: {e}")

    # Dataframe for ImageName & Check Extracted Check Info
    def create_dataframe_from_folder(folder_path): 
        df_list = []
            
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                file_path = os.path.join(folder_path, txt_filename)

                if os.path.exists(file_path):
                    result = process_check(image_path, file_path)
                    df_list.append({'Image Name': filename, 'check_info': result})

        result_df = pd.DataFrame(df_list)
        return result_df

    # Reject letter Classification model 
    def load_and_predict_text_classification_model(reject_model_path, tokenizer, max_len, df,label_encoder):
        
        # Set the logging verbosity level to suppress info and below
        tf.get_logger().setLevel('ERROR')
        model = load_model(reject_model_path)
        predictions_dict = {}
    
        for index, row in df.iterrows():           
            image_path = row['reason_image_path']           
           # Check if the image is loaded successfully
            if image is None:
                print(f"Error: Unable to load image: {image_path}")
                continue

            image_text = pytesseract.image_to_string(cv2.imread(image_path), lang='eng').strip()
            new_texts_seq = tokenizer.texts_to_sequences([image_text])
            new_texts_pad = pad_sequences(new_texts_seq, maxlen=max_len, padding='post')
            prediction = model.predict(new_texts_pad)
            decoded_predictions = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
            predictions_dict[image_path] = decoded_predictions[0]
            
        df['reject_reason'] = df['reason_image_path'].map(predictions_dict)
        return df

    # Extraction from pdf check text
    def check_text_extraction(pdf_path, image_folder):
        # Check if the file has a PDF extension
        if pdf_path.lower().endswith(".pdf"):
            
            # Create a dictionary to store the details for each page
            all_details = {}
    
            # Get the base filename (excluding the extension) from the PDF path
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
            # Open the PDF file in binary mode
            with open(pdf_path, 'rb') as pdf_file_obj:
                
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    
                # Iterate through each page in the PDF
                for page_number in range(len(pdf_reader.pages)):
                    
                    # Get the page object
                    page_obj = pdf_reader.pages[page_number]
    
                    # Extract text from the page
                    text = page_obj.extract_text()
    
                    # Define the regex pattern
                    account_number_pattern = re.compile(r'Account Number: (\d+)')
                    # check_number_pattern = re.compile(r'Check Number: (\d+)')
                    transaction_pattern = re.compile(r'Transaction: (\d+)')
                    lockbox_pattern = re.compile(r'Lockbox: (\d+)')
                    batch_pattern = re.compile(r'Batch: (\d+)')
    
                    # Find the match in the text
                    match1 = account_number_pattern.search(text)
                    # match2 = check_number_pattern.search(text)
                    match3 = transaction_pattern.search(text)
                    match4 = lockbox_pattern.search(text)
                    match5 = batch_pattern.search(text)
    
                    # Extract the details if a match is found
                    if match1 or match3 or match4 or match5:
                        account_number = match1.group(1) if match1 else None
                        # check_number = match2.group(1) if match2 else None
                        Transaction = match3.group(1) if match3 else None
                        Lockbox = match4.group(1) if match4 else None
                        Batch = match5.group(1) if match5 else None
    
                        # Create a dictionary to store the details
                        details_dict = {
                            "Subscriber Account Number": account_number,
                            # "Check Number": check_number,
                            "Transaction": Transaction,
                            "Lockbox": Lockbox,
                            "Batch": Batch
                        }
    
                        # Use the page number as the key in the all_details dictionary
                        page_name = f"{filename}_page{str(page_number + 1).zfill(3)}.png"
    
                        # Create the image filename using the specified format
                        image_filename = os.path.join(image_folder, page_name)
    
                        # print(f"Saving image: {image_filename}")
                        all_details[image_filename] = details_dict                    
        
                    else:
                        
                        # If no details are found, store an empty dictionary
                        page_name = f"{filename}_page{str(page_number + 1).zfill(3)}.png"
    
                        # Create the image filename using the specified format
                        image_filename = os.path.join(image_folder, page_name)
                        
                        all_details[image_filename] = {}
                        
            # Print or return the all_details dictionary as needed
            return all_details
        
    # Process data and return JSON
    def process_data():

        # pdf to Image Conversion
        extract_images_from_pdfs(pdf_path, image_folder)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # Image Classification into checks & letters
        Check_Reason_dataframe = process_classification_results(image_folder,document_folder)

        # Object Detection on checks
        detection_results = run_detection_on_dataframe(Check_Reason_dataframe)

        # df for Image Name with Extarcted check Info
        result_dataframe = create_dataframe_from_folder(detection_folder)

        # Extraction from pdf check text
        all_details = check_text_extraction(pdf_path, image_folder)
        
        # Merging both dataframes 
        merged_df = pd.merge(Check_Reason_dataframe, result_dataframe, left_on='check_name', right_on='Image Name', how='left')

        # Dropping Image name column from df
        merged_df = merged_df.drop('Image Name', axis=1)

        # Reject letter Classification Model
        df = load_and_predict_text_classification_model(reject_model_path, tokenizer, max_len, merged_df,label_encoder)

        # Dropping Image name column from df
        df = df.drop('check_name', axis=1)

        desired_columns = ['check_image_path', 'check_info', 'reason_image_path', 'reject_reason']
        # Re-ordering the sequence
        df = df[desired_columns]

        # For mapping extracted text from bottom of check with the check Image Info
        for index, row in df.iterrows():
                check_image_path = row['check_image_path']
                if check_image_path in all_details:
                    details = all_details[check_image_path]
                    current_check_info = row['check_info']
                    current_check_info.update(details)
                    df.at[index, 'check_info'] = current_check_info
                    
        
        json_data = df.to_json(orient='records', lines=True)

        # Parse JSON string to a Python object (list of dictionaries in this case)
        data = [json.loads(line) for line in json_data.split('\n') if line.strip()]
        
        # Create the desired dictionary structure
        result_dict = {'response': data}
    
        # Convert the dictionary to a formatted JSON string
        formatted_json = json.dumps(result_dict, indent=2)
        
        # Replace double quotes with single quotes and double backslashes with single slashes
        json_string_modified = formatted_json.replace('"', "'").replace('\\\\', '\\')
        print(json_string_modified)
        # return json_string_modified
        
    return process_data()