--AZURE

Connect-AzAccount

--azure registry

#Create resource group and then containers registry
az login
az group create --name RG-ACR --location eastus
az acr create --resource-group RG-ACR --name mywebappacrdemo --sku Standard

#Push to registry
docker tag accenture_hack-next bartfastap1234.azurecr.io/containergroup:next
docker tag accenture_hack-fastapi bartfastap1234.azurecr.io/containergroup:fastapi

#Login to containers registry
docker login azure

az acr login --name bartfastap1234

#Push images
docker push bartfastap1234.azurecr.io/containergroup:next
docker push bartfastap1234.azurecr.io/containergroup:fastapi

#Create Azure Context for container instances to deploy them
docker login azure
docker context create aci hack_context

#Uzycie utworzonego kontekstu
docker context use hack_context

#Deploy
docker compose up

--SQL

mysql -u root -p

--dockerfiles

docker run -p 8000:8000 fastapi
docker run -p 3000:3000 next
