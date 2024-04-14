# Use an updated version of python:3.10-slim-bullseye which might have the vulnerability fixed
FROM python:3.10-slim-bullseye AS base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN useradd -m appuser

RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    curl \
    gnupg \
    unixodbc-dev \
    libgssapi-krb5-2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="$PATH:/opt/mssql-tools18/bin"

USER appuser
COPY requirements.txt .

USER root
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

USER appuser
COPY --chown=appuser:appuser . .

# Verify that uvicorn is installed
RUN which uvicorn || (echo "Uvicorn not found" && exit 1)

# Expose the port that the app runs on
EXPOSE 8000

# Use a non-root user to run the application
USER appuser

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
