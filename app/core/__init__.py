"""Business logic: the state machine, query understanding, retrieval, the
honesty guardrails, narrative assembly, sessions, and the closed vocabulary.

The response contract (`core.schemas`) lives here too, because it is the domain
boundary the service produces and the API merely serialises. `core` may use
`infrastructure` for persistence (models, db session) but never imports `api`.
"""
