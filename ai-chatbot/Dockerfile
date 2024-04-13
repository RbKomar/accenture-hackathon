FROM node:18-alpine

# Create work directory
WORKDIR /usr/src/app

# Install runtime dependencies
# RUN npm install yarn -g

# Copy app source to work directory
COPY . /usr/src/app

# Install app dependencies
RUN npm install

# ARG PORT=9005
# ENV NODE_ENV=$ENV
# ENV APP_ENV=$APP_ENV
# ENV PORT=$PORT



# USER nextjs
EXPOSE 3000

# Build and run the app
# EXPOSE $PORT
RUN npm run build
CMD npm run start
