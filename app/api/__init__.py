"""HTTP layer: routers, request/response wiring, auth, middleware, error shape.

Depends on `core` (business logic) and `infrastructure` (db session, metrics).
Nothing in `core` or `infrastructure` may import from here.
"""
