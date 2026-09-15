"""A wall-clock deadline guard; revocation does not imply HTTP cancellation."""
from time import monotonic


class DeadlineModelClient:
    def __init__(self, client, deadline_wall_s, *, clock=monotonic):
        self._client = client
        self._deadline = float(deadline_wall_s)
        self._clock = clock

    @property
    def model(self):
        return getattr(self._client, "model", None)

    def chat(self, messages, *, options=None):
        if self._clock() >= self._deadline:
            raise TimeoutError("runtime model pipeline deadline expired before HTTP")
        response = self._client.chat(messages, options=options)
        if self._clock() >= self._deadline:
            raise TimeoutError("runtime model pipeline deadline expired during HTTP")
        return response
