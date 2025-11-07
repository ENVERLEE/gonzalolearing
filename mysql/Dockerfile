FROM mysql:8.0

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD mysqladmin ping -h localhost || exit 1

EXPOSE 3306

