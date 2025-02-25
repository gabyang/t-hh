### Healthhacks
Application aims to integrate the go-to-market strategy into the data processing pipelines of this repository.

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

5) Create an .env file with the OPENAI_API_KEY key