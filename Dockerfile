# Use a base Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker's layer caching
COPY requirements.txt /app/requirements.txt

# Update pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install the required Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose the port that the Flask app will run on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "model.py"]
