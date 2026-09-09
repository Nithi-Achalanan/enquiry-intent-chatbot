# Dependency and Failure Map

Both agents synchronously call Groq. Timeout, connection failure, 429, and 5xx are retried within a bounded budget; other failures fail immediately.
