# Use a base image with both Python and Java
FROM openjdk:11-jre-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python and necessary packages
RUN apt-get update && apt-get install -y python3 python3-pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask's development server
EXPOSE 5000

# Run app.py when the container launches using Flask's built-in server
CMD ["python3", "app.py"]
