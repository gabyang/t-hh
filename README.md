### Healthhacks
Application aims to integrate the go-to-market strategy into the data processing pipelines of this repository.
## Setup

Steps to quickstart:
1) Install IRIS Community Edtion in a container.
```
docker run -d --name iris-comm -p 1972:1972 -p 52773:52773 -e IRIS_PASSWORD=demo -e IRIS_USERNAME=demo intersystemsdc/iris-community:latest
```

2) Create a Python environment and activate it. Use versions between 3.8 to 3.12

3) Install requirements.txt 
```
pip install -r requirements.txt
```

4) Retrieve the DB API driver from the [Intersystems Github repository](https://github.com/intersystems-community/hackathon-2024) under the folder called "install" and ensure that it matches with your Operating System

5) Create an .env file with:
```
OPENAI_API_KEY=
TELEGRAM_BOT_TOKEN=
```

## Notes

- Ensure good lighting conditions for accurate measurements
- Keep your face clearly visible in the frame
- Stay still while taking photos/videos
- Results are estimates and should not be used for medical purposes

## Technical Details

The bot uses computer vision techniques to:
1. Detect faces in images/videos
2. Extract color signals from the face region
3. Process these signals using FFT to estimate heart rate
4. Calculate blood oxygen levels using color information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Vital Signs Monitor Telegram Bot
This Telegram bot analyzes photos or videos of faces to estimate vital signs including heart rate and blood oxygen levels using remote photoplethysmography (rPPG) techniques.

## Features

- Heart rate estimation
- Blood oxygen level (SpO2) estimation
- Supports both photo and video inputs
- Real-time face detection and processing
- User-friendly interface with emoji feedback


To get a bot token:
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Use the `/newbot` command and follow the instructions
3. Copy the token provided by BotFather