# Failure Mode Register

P1: transient Groq timeout/rate-limit/server error previously surfaced as generic 500 with no structured evidence. Containment: 30-second request timeout, two bounded retries, sanitized 503 after exhaustion.
