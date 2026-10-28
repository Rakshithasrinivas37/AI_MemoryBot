"""Exercise packaged frontend routes without model downloads or API keys."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


class ContainerAPITests(unittest.TestCase):
    def setUp(self):
        bot = ModuleType('Chatbot.bot')
        bot.Chatbot = Mock()
        evaluation = ModuleType('Evaluation.evaluate')
        evaluation.Evaluate = Mock()
        with patch.dict(sys.modules, {'Chatbot.bot': bot, 'Evaluation.evaluate': evaluation}):
            spec = importlib.util.spec_from_file_location('container_api_test', ROOT / 'Backend/app.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.client = TestClient(module.app)
        self.addCleanup(self.client.close)

    def test_health_does_not_require_a_session(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

    def test_frontend_and_assets_are_served_from_api_origin(self):
        for path, expected in [('/', 'MemoryBot'), ('/chat.html', 'memory-detail'),
                               ('/api-config.js', 'MEMORYBOT_API_BASE'), ('/style.css', '.chat-main')]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(expected, response.text)

    def test_static_mount_does_not_shadow_api_or_expose_source(self):
        self.assertEqual(self.client.get('/memories?username=unknown').status_code, 404)
        self.assertEqual(self.client.get('/memories?username=unknown').json()['message'], 'User not found')
        self.assertEqual(self.client.get('/Backend/app.py').status_code, 404)
        self.assertEqual(self.client.get('/.env').status_code, 404)
