# Stage 1: Build the Go binary
FROM golang:1.22-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Download Go modules (caches dependencies if mod/sum haven't changed)
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the source code
COPY . .

# Build the Go application statically
# CGO_ENABLED=0 ensures the binary doesn't rely on system C libraries
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /gateway-binary ./cmd/main.go

# Stage 2: Create the minimal production image
FROM alpine:3.19

# Install CA certificates so the gateway can make secure HTTPS calls to the JWKS URL
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the compiled binary from the builder stage
COPY --from=builder /gateway-binary .

# Ensure the binary is executable
RUN chmod +x ./gateway-binary

# Expose the default port
EXPOSE 8080

# Command to run the executable
CMD ["./gateway-binary"]