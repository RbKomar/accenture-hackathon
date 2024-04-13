# Use the official Python image as a base
# FROM python:3.9-slim
FROM python:3.9-bullseye

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Avoid prompts from apt by setting noninteractive environment
ENV DEBIAN_FRONTEND=noninteractive

# Add Microsoft's GPG key
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc

# Add the Microsoft SQL Server repository
RUN curl https://packages.microsoft.com/config/debian/11/prod.list | tee /etc/apt/sources.list.d/mssql-release.list

# Update the package lists
RUN apt-get update

# Install SQL Server ODBC Driver
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Optional: Install SQL Server Command Line Tools
RUN ACCEPT_EULA=Y apt-get install -y mssql-tools18

# Optional: Install unixODBC development headers
RUN apt-get install -y unixodbc-dev

# Optional: Install Kerberos library for Debian-slim distributions
RUN apt-get install -y libgssapi-krb5-2

# Clean up to keep the image clean and small
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Add SQL tools to the PATH
RUN echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> /root/.bashrc

# Copy the current directory contents into the container at /app
COPY . .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
