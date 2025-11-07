package com.leareng.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.SQLException;

/**
 * Database connection health check on startup
 * Waits for database to be ready before proceeding
 */
@Component
@Order(1)
public class DatabaseConfig implements CommandLineRunner {
    
    private static final Logger logger = LoggerFactory.getLogger(DatabaseConfig.class);
    
    @Autowired
    private DataSource dataSource;
    
    @Autowired(required = false)
    private JdbcTemplate jdbcTemplate;
    
    @Override
    public void run(String... args) {
        waitForDatabase();
    }
    
    private void waitForDatabase() {
        int maxRetries = 30;
        int retryDelay = 2000; // 2 seconds
        
        logger.info("Checking database connection...");
        
        for (int i = 0; i < maxRetries; i++) {
            try (Connection connection = dataSource.getConnection()) {
                if (connection.isValid(5)) {
                    // Test query
                    if (jdbcTemplate != null) {
                        jdbcTemplate.queryForObject("SELECT 1", Integer.class);
                    }
                    logger.info("Database connection successful!");
                    return;
                }
            } catch (SQLException e) {
                if (i < maxRetries - 1) {
                    logger.warn("Database connection attempt {}/{} failed. Retrying in {} ms... Error: {}", 
                        i + 1, maxRetries, retryDelay, e.getMessage());
                    try {
                        Thread.sleep(retryDelay);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while waiting for database", ie);
                    }
                } else {
                    logger.error("Failed to connect to database after {} attempts", maxRetries);
                    throw new RuntimeException("Database connection failed after retries: " + e.getMessage(), e);
                }
            } catch (Exception e) {
                if (i < maxRetries - 1) {
                    logger.warn("Database check attempt {}/{} failed. Retrying in {} ms...", 
                        i + 1, maxRetries, retryDelay);
                    try {
                        Thread.sleep(retryDelay);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted while waiting for database", ie);
                    }
                } else {
                    logger.error("Failed to connect to database after {} attempts", maxRetries);
                    throw new RuntimeException("Database connection failed: " + e.getMessage(), e);
                }
            }
        }
    }
}

