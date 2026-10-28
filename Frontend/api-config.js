// The packaged application serves UI and API from the same origin.
// serve.py overrides this file for the separate local frontend server.
window.MEMORYBOT_API_BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';
