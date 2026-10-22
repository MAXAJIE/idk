# Architecture
- Keep execution local in Python and the spectator API read-only; credentials must not enter browser responses.
- Resolve CLI defaults from the backend directory so root and backend launches share configuration and storage.
- LLM output explains persisted decisions only and must never alter qualification, risk gates or orders.
- Reject unqualified order execution; allow explicit unqualified selection only in dry-run research.