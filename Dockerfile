FROM python:latest

# load data
COPY . /app/

# change cwd to app
WORKDIR /app

# install requirements
RUN pip install --no-cache-dir -r requirements.txt

# make entrypoint executable
RUN chmod +x entrypoint.sh

EXPOSE 80

# excute the entrypoint.sh file
ENTRYPOINT ["./entrypoint.sh"]