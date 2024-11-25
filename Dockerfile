# Use a base Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the necessary files and folders
COPY . /app

# Expose the port that the Flask app will run onmm
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "model.py"]
