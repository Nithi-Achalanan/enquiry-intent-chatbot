# Decisions

Selected application-owned retries over SDK defaults so attempts are observable and bounded. Retrying non-transient errors is rejected because it cannot recover auth, validation, or schema failures.
