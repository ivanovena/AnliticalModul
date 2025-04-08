FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json ./
COPY postcss.config.js ./
COPY tailwind.config.js ./
RUN npm install --legacy-peer-deps
RUN npm install ajv autoprefixer postcss tailwindcss

# Copy application code
COPY public/ ./public/
COPY src/ ./src/

# Create an empty .env file to avoid warnings
RUN touch .env

# Build the application with debugging output
RUN echo "Building application..." && \
    npm run build || { echo "Build failed"; exit 1; }

# Production stage
FROM nginx:alpine

# Copy the build output
COPY --from=build /app/build /usr/share/nginx/html

# Copy the nginx configuration
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Create basic favicon to prevent 404 errors
# Create a healthcheck file
RUN echo "OK" > /usr/share/nginx/html/healthcheck.html

# Healthcheck to make sure nginx is running and responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --spider -q http://localhost/healthcheck.html || exit 1

# Expose port
EXPOSE 80

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
