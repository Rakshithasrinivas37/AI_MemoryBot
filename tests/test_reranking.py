"""Reranker tests without model downloads or inference dependencies."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class RerankingTests(unittest.TestCase):
    def setUp(self):
        package = ModuleType('sentence_transformers')
        self.model = Mock()
        self.constructor = Mock(return_value=self.model)
        package.CrossEncoder = self.constructor
        spec = importlib.util.spec_from_file_location(
            'reranker_under_test', Path(__file__).resolve().parents[1] / 'RAG' / 'llm_reranking.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'sentence_transformers': package}):
            spec.loader.exec_module(self.module)

    def test_scores_sort_descending_and_model_is_reused(self):
        self.model.predict.return_value = [-2.0, 3.0, 0.5]
        memories = ['low', 'high', 'middle']
        for _ in range(2):
            self.assertEqual(self.module.llm_rerank('question', memories), ['high', 'middle', 'low'])
        self.constructor.assert_called_once_with(
            'cross-encoder/ms-marco-MiniLM-L6-v2', device='cpu', max_length=512)
        self.model.predict.assert_called_with(
            [('question', m) for m in memories], show_progress_bar=False)
        self.assertEqual(memories, ['low', 'high', 'middle'])

    def test_trivial_inputs_skip_model_loading(self):
        self.assertEqual(self.module.llm_rerank('question', []), [])
        self.assertEqual(self.module.llm_rerank('question', ['only']), ['only'])
        self.constructor.assert_not_called()

    def test_ties_and_duplicates_preserved(self):
        self.model.predict.return_value = [1, 1, 1]
        self.assertEqual(self.module.llm_rerank('question', ['a', 'b', 'a']), ['a', 'b', 'a'])

    def test_inference_failure_preserves_hybrid_order(self):
        self.model.predict.side_effect = RuntimeError('inference unavailable')
        with self.assertLogs(self.module.logger, level='WARNING'):
            self.assertEqual(self.module.llm_rerank('question', ['a', 'b']), ['a', 'b'])

    def test_invalid_scores_preserve_hybrid_order(self):
        for scores in [[1], [float('nan'), 2], [float('inf'), 2]]:
            with self.subTest(scores=scores), self.assertLogs(self.module.logger, level='WARNING'):
                self.model.predict.return_value = scores
                self.assertEqual(self.module.llm_rerank('question', ['a', 'b']), ['a', 'b'])

    def test_legacy_client_is_not_used(self):
        client = Mock()
        self.model.predict.return_value = [0, 1]
        self.assertEqual(self.module.llm_rerank('question', ['a', 'b'], client), ['b', 'a'])
        self.assertEqual(client.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
