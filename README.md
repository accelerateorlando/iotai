To get this running on a laptop...


#1 - make sure you set up your video camera to stream to docker - I had to do some WSL stuff to make that happen

usbipd list
usbipd bind --busid 2-3
sudo apt install v4l-utils

#2 - run the docker image

docker compose build
docker compose up -d

Your camera should light up and you should see MQTT messages in the log.  You may need to update docker-compose.yml if you see error messages with MQTT / etc.