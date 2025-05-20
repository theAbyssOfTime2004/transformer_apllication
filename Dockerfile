# Use an official Python runtime as a parent image.
# python:3.9-slim is a good choice for a smaller image size.
FROM python:3.9-slim

# Set the working directory in the container to /app.
# All subsequent commands (like COPY, RUN, CMD) will be executed from this directory.
WORKDIR /app

# Copy the requirements.txt file from the host to the container's /app directory.
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt.
# --no-cache-dir: Disables the pip cache, which can help reduce the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code from the host's current directory (.)
# into the container's /app directory.
# This includes your app.py, the 'templates' directory,
# your 'transformer_text_classification' package, and importantly,
# the 'saved_model_and_tokenizer' directory.
COPY . .

# Inform Docker that the container will listen on port 5000 at runtime.
# This does not actually publish the port; it's more of a documentation step
# and a hint to the user running the container.
EXPOSE 5000

# Command to run the application when the container starts.
# We use Gunicorn as the WSGI server for the Flask application.
#   --workers 2: Specifies the number of worker processes. Adjust as needed (e.g., (2 * CPU cores) + 1).
#   --bind 0.0.0.0:5000: Binds Gunicorn to all network interfaces on port 5000 within the container.
#   app:app: Tells Gunicorn to look for a Flask application instance named 'app' in a Python module named 'app.py'.
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]