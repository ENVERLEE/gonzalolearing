FROM mysql:8.0

# Install Python for simple HTTP health check server
RUN apt-get update && \
    apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/*

# Create a simple HTTP health check server
RUN echo '#!/usr/bin/env python3\n\
import http.server\n\
import socketserver\n\
import subprocess\n\
import time\n\
import sys\n\
\n\
class HealthCheckHandler(http.server.BaseHTTPRequestHandler):\n\
    def do_GET(self):\n\
        try:\n\
            # Check MySQL status\n\
            result = subprocess.run(\n\
                ["mysqladmin", "ping", "-h", "localhost"],\n\
                capture_output=True,\n\
                timeout=5\n\
            )\n\
            if result.returncode == 0:\n\
                self.send_response(200)\n\
                self.send_header("Content-type", "text/plain")\n\
                self.end_headers()\n\
                self.wfile.write(b"OK")\n\
            else:\n\
                self.send_response(503)\n\
                self.send_header("Content-type", "text/plain")\n\
                self.end_headers()\n\
                self.wfile.write(b"Service Unavailable")\n\
        except Exception as e:\n\
            self.send_response(503)\n\
            self.send_header("Content-type", "text/plain")\n\
            self.end_headers()\n\
            self.wfile.write(b"Error")\n\
    \n\
    def log_message(self, format, *args):\n\
        pass  # Suppress logging\n\
\n\
if __name__ == "__main__":\n\
    # Wait for MySQL to be ready before starting server\n\
    print("Waiting for MySQL to be ready...", file=sys.stderr)\n\
    for i in range(60):\n\
        try:\n\
            result = subprocess.run(\n\
                ["mysqladmin", "ping", "-h", "localhost"],\n\
                capture_output=True,\n\
                timeout=2\n\
            )\n\
            if result.returncode == 0:\n\
                print("MySQL is ready! Starting health check server...", file=sys.stderr)\n\
                break\n\
        except:\n\
            pass\n\
        time.sleep(1)\n\
    \n\
    PORT = 8080\n\
    with socketserver.TCPServer(("", PORT), HealthCheckHandler) as httpd:\n\
        httpd.serve_forever()\n\
' > /healthcheck.py && chmod +x /healthcheck.py

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Start health check server in background\n\
python3 /healthcheck.py &\n\
# Use MySQL default entrypoint\n\
exec /usr/local/bin/docker-entrypoint.sh mysqld\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 3306 8080

ENTRYPOINT ["/entrypoint.sh"]

