#Note to users: I have commented the file paths, so in order for the code to work, you must enter it's specified file path yourself


import speech_recognition as sr
import subprocess
import elevenlabs
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from playsound import playsound
import os
import time
import sys
import ctypes

#Imports for AI
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np

os.environ["EMNIST_CACHE_DIR"] = #Enter path to EMIST folder

def elevate_to_admin():
    if ctypes.windll.shell32.IsUserAnAdmin():
        return True
    else:
        playsound(#Enter path to get_admin.mp3 in cache audio folder#)
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        playsound(#Enter path to got_admin.mp3 in cache audio folder#)
        sys.exit()
        

time_limit = 15

client = ElevenLabs(api_key = #Enter Elvenlabs API key#)

recognizer = sr.Recognizer()

ding_audio_path = #Enter path to ding-sound-246413.mp3 in cache audio folder
ffplay_path = #Enter path to ffplay.exe


#----------This section of the code trains, create and loads the AI, if the code letter recogntition AI does not work, uncomment this section to train the AI-------------------------------------------------#

#Note: If the loading EMNIST dataset doesn't work, you will have to manually download the dataset
#Loading the EMNIST dataset
#def load_data_set():
    #x_train, y_train = extract_training_samples("letters")
    #x_test, y_test = extract_test_samples("letters")

    #x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255.0
    #x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255.0

    #y_train = to_categorical(y_train - 1, 26)
    #y_test = to_categorical(y_test -1, 26)

    #return x_train, y_train, x_test, y_test

#CN model
#def create_model():
    #model = Sequential([
        #Input(shape=(28, 28,1)),
        #Conv2D(32, (3, 3), activation='relu'),
        #MaxPooling2D((2, 2)),
        #Dropout(0.25),
        #Conv2D(64, (3, 3), activation='relu'),
        #MaxPooling2D((2, 2)),
        #Dropout(0.25),
        #Flatten(),
        #Dense(128, activation='relu'),
        #Dropout(0.5),
        #Dense(26, activation='softmax')
    #])
    #model.compile(optimizer='adam',
                  #loss='categorical_crossentropy',
                  #metrics=['accuracy'])
    #return model

#Training & Saving the model
#def train_model():
    #x_train, y_train, x_test, y_test = load_data_set()
    #model = create_model()

    #checkpoint_filepath = "handwriting_model_checkpoint.h5"

    #if os.path.exists(checkpoint_filepath):
        #print("Resuming from checkpoint....")
        #model = load_model(checkpoint_filepath)
    #else:
        #model = create_model()

    #checkpoint_called = ModelCheckpoint(
        #checkpoint_filepath,
        #save_best_only = True,
        #save_weights_only = False,
        #monitor = "val_loss",
        #mode = "min",
        #verbose = 1
    #)

    #try:
        #model.fit(
            #x_train,
            #y_train,
            #validation_data = (x_test, y_test),
            #epochs = 10,
            #batch_size = 128,
            #verbose = 1,
            #callbacks = [checkpoint_called]
        #)

    #except KeyboardInterrupt:
        #print("Training interrupted. Saving model...")
        #model.save("handwritting_model_interrupted.h5")

    #model.save("handwriting_model.h5")
    #print("Model trained and saved as handwriting_model.h5")

#train_model()

#Importing & Load the model
#from tensorflow.keras.models import load_model
#import cv2
#import numpy as np

#handwriting_model = load_model("handwriting_model.h5")

#def preprocess_image(image_path):
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #img = cv2.resize(img, (28, 28))
    #img = img / 255.0
    
    #img = np.expand_dims(img, axis=(0, -1))
    #return img

#def recognise_handwriting(image_path):
    #processed_img = preprocess_image(image_path)
    #prediction = handwriting_model.predict(processed_img)
    #predicted_label = chr(np.argmax(prediction) + 65)
    #return predicted_label
 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#Voice commands
def speak(text_chunk):
    audio = client.text_to_speech.convert(
    text=text_chunk,
    voice_id="CQcj2MsUgZyAgfHH6yJV",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
    )
    
    audio_data = b"".join(audio)

    audio_file_path = "audio.mp3"
    with open(audio_file_path, "wb") as f:
        f.write(audio_data)
        
    try:
        subprocess.run([ffplay_path, "-autoexit", "-nodisp", audio_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while playing the audio: {e}")

    time.sleep(0.5)


def listen(time_limit):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, phrase_time_limit=time_limit)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Unknown"
    except sr.RequestError as e:
        return "conection error"


def change_time_limit(time_limit):
    playsound(#path to current_time_limit.mp3 in cache audio folder#)
    speak(str(time_limit))
    playsound(#Enter path to change_time_limit.mp3 in cache audio folder#)
    new_time_limit = int(input("Enter the new time limit: "))
    try:
        time_limit = int(new_time_limit)
        playsound(#Enter path to time_limit_changed.mp3 in cache audio folder#)
        speak(str(time_limit))
    except ValueError:
        playsound(#Enter path to time_limit_not_number.mp3 in chace audio folder#)

#Calender and Reminders
from tkinter import *
import datetime

def reminder():
    now = datetime.datetime.now()
    if now.strftime("%Y-%m-%d") == "2024-12-31":
        speak("Reminder: New Year's Eve party today!")

#Web Automation
import webbrowser

#Note: I used opergx here for personal perferences, you can use any web browser that you want
webbrowser.register('opera_gx', None, webbrowser.BackgroundBrowser(#Enter path to oper gx#))

def open_google():
    webbrowser.get('opera_gx').open("https://www.google.com")

def open_youtube():
    webbrowser.get('opera_gx').open("https://www.youtube.com")

#File Access
import os

def open_file(path):
    os.startfile(path)

#Admin Access
import ctypes

def run_as_admin():
    if ctypes.windll.shell32.IsUserAnAdmin():
        return True
    else:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", "python", __file__, None, 1)

#PC Monitoring
import psutil

def check_system():
    cpu = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage('/')
    speak(f"CPU usage is {cpu} percent. Disk usage is {disk.percent} percent.")

#Chatbot
import json
from difflib import get_close_matches

def load_knowledge_base(file_path: str) -> dict:
    if getattr(sys, 'frozen', False):
        file_path = os.path.join(sys._MEIPASS, file_path)
    
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data

def save_knowledge_base(file_path: str, data: dict):
    if getattr(sys, 'frozen', False):
        file_path = os.path.join(sys._MEIPASS, file_path)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.7)
    return matches[0] if matches else None

def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]

def chat_bot(command):
    knowledge_base: dict = load_knowledge_base('knowledge_base.json')
    loop = False

    best_match: str | None = find_best_match(command, [q["question"] for q in knowledge_base["questions"]])

    if best_match:
        answer: str = get_answer_for_question(best_match, knowledge_base)
        speak(answer)
    else:
        loop = True
        while loop == True:
            playsound(#Enter path to Don't_know_answer_to_question.mp3 in cache audio folder#)
            yes_or_no = listen(time_limit)

            if yes_or_no.lower() == "yes" or yes_or_no.lower() == "sure":
                while loop == True:
                    playsound(#Enter path to Say_answer_now.mp3 in cache audio folder#)
                    new_answer = listen(time_limit)
                
                    if new_answer.lower() != "unknown":
                        knowledge_base["questions"].append({"question": command, "answer": new_answer})
                        save_knowledge_base('knowledge_base.json', knowledge_base)
                        playsound(#Enter path to Learned_new_response.mp3 in cache audio folder#)
                        loop = False
                    else:
                        playsound(#Enter path to error_understand.mp3 in cache audio folder#)
            elif yes_or_no.lower() == "no" or "no" in yes_or_no.lower():
                playsound(#Enter path to Question_&_answer_not_recorded.mp3 in cache audio folder#)
                loop = False
            else:
                playsound(#Enter path to error_understand.mp3 in cache audio folder#)
                
        



#Opening
from datetime import datetime
def greeting():
    now = datetime.now()
    current_hour = now.hour

    if 5 <= current_hour < 12:
        playsound(#Enter path to morning.mp3 in cache audio folder#)
        playsound(#Enter path to help.mp3 in cache audio folder#)
    elif 12 <= current_hour < 17:
        playsound(#Enter path to afternoon.mp3 in cache audio folder#)
        playsound(#Enter path to help.mp3 in cache audio folder#)
    elif 17 <= current_hour < 21:
        playsound(#Enter path to evening.mp3 in cache audio folder#)
        playsound(#Enter path to help.mp3 in cache audio folder#)
    else:
        playsound(#Enter path to greeting.mp3 in cache audio folder#)
        playsound(#Enter path to help.mp3 in cache audio folder#)

#elevate_to_admin()

greeting()
    


#Loop
game_mode = False
voice_key = False


if __name__ == "__main__":
    while True:
        activation = listen(time_limit)
        if activation.lower() == "jarvis":
            print("Yes?")
            playsound(ding_audio_path)
            voice_key = True
        

        while voice_key == True:
            command = listen(time_limit)

            if command.lower() == "unknown":
                print("You said", command)
                playsound(#Enter path to error_understand.mp3 in cache audio folder#)
                voice_key = False
            elif command.lower() == "conection error":
                playsound(#Enter path to Connect_issue_with_API.mp3 in cache audio folder#)
                voice_key = False
            elif "google" in command.lower():
                open_google()
                voice_key = False
            elif "youtube" in command.lower():
                open_youtube()
                voice_key = False
            elif "reminder" in command.lower():
                reminder()
                voice_key = False
            elif "check system" in command.lower():
                check_system()
                voice_key = False
            elif "exit" in command.lower() or "close" in command.lower() or "shut down" in command.lower():
                playsound(#Enter path to goodbye.mp3 in cache audio folder#)
                voice_key = False
                sys.exit()
            elif "time limit" in command.lower():
                change_time_limit(time_limit)
                voice_key = False
            elif "scan" in command.lower() and "letter" in command.lower():
                speak("Enter the image path in the console.")
                image_path = input("Enter the image path here: ")
                if image_path:
                    recognised_character = recognise_handwritten(image_path)
                    speak(f"The recognised character is {recognised_character}.")
                    voice_key = False
                else:
                    speak("Error, No image found.")
                    voice_key = False
            else:
                print(f"You said: {command}")
                chat_bot(command)
                voice_key = False
















