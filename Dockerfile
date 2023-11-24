# Use the official Python image as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Remove temporary files
RUN rm -rf /app/temporary_folder && rm -rf /var/lib/apt/lists/* 
# Install any dependencies
RUN  apt-get update  && pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run your application
CMD ["streamlit", "run", "app.py"]