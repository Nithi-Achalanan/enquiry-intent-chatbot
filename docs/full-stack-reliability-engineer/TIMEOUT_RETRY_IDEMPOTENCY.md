# Timeout and Retry

ChatGroq has a 5-second connect and 30-second request timeout. The application owns two retries with exponential backoff and jitter; SDK retries are disabled to prevent retry amplification. Chat requests are read-only and safe to retry.
