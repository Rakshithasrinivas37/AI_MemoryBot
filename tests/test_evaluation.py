import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.create = Mock()
        groq = ModuleType('groq')
        groq.Groq = Mock(return_value=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))))
        spec = importlib.util.spec_from_file_location('eval_test', Path(__file__).resolve().parents[1]/'Evaluation/evaluate.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'groq':groq}):
            spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.module.LOG_PATH = Path(self.temp.name)/'eval.json'
        self.scores = {'correctness':None,'groundedness':None,'relevance':4,'completeness':3,
                       'instruction_compliance':4,'verification':'cannot_verify','reason':'No supporting evidence.'}
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.scores)))])

    def test_context_is_sent_and_unknown_scores_logged(self):
        context = {'conversation':[{'role':'user','content':'I live in Singapore'}],
                   'memories':'User is an engineer', 'tool_results':[{'result':'source snippet'}],
                   'current_date':'October 26, 2026', 'instructions':'Answer briefly'}
        evaluator = self.module.Evaluate('key','What about here?','An answer',context)
        context['conversation'].clear()
        self.assertEqual(evaluator.auto_evaluate(),self.scores)
        payload = json.loads(self.create.call_args.kwargs['messages'][1]['content'])
        self.assertEqual(payload['conversation'][0]['content'],'I live in Singapore')
        self.assertEqual(payload['memories'],'User is an engineer')
        self.assertEqual(payload['tool_results'],[{'result':'source snippet'}])
        self.assertEqual(json.loads(self.module.LOG_PATH.read_text())[0]['scores']['correctness'],None)

    def test_contextual_advice_can_have_scores_without_factual_verification(self):
        scores = dict(self.scores, correctness=4, groundedness=5, verification='not_applicable',
                      reason='Practice questions fit the previously discussed AI Engineer role; no confirmed employer questions are claimed.')
        self.create.return_value.choices[0].message.content = json.dumps(scores)
        context = {'conversation': [{'role': 'user', 'content': 'I am preparing for an AI Engineer interview focused on RAG.'}],
                   'memories': 'User works with Python', 'tool_results': []}
        result = self.module.Evaluate('key', 'What questions should I expect?',
                                      'Practice explaining retrieval, reranking, and evaluating RAG.', context).auto_evaluate()
        self.assertEqual(result, scores)
        payload = json.loads(self.create.call_args.kwargs['messages'][1]['content'])
        self.assertEqual(payload['conversation'], context['conversation'])
        self.assertEqual(payload['tool_results'], [])
        self.assertEqual(json.loads(self.module.LOG_PATH.read_text())[0]['scores']['verification'], 'not_applicable')

    def test_answer_truncation_is_explicit_in_evaluator_payload(self):
        evaluator = self.module.Evaluate('key', 'Explain RAG', 'Detail. ' * 1200)
        evaluator.auto_evaluate()
        payload = json.loads(self.create.call_args.kwargs['messages'][1]['content'])
        self.assertEqual(payload['truncated_fields'], ['answer'])
        self.assertTrue(payload['answer'].endswith('[TRUNCATED]'))
        self.assertGreater(len(evaluator.response), len(payload['answer']))

    def test_api_failure_returns_none(self):
        self.create.side_effect = RuntimeError('rate limited')
        with self.assertLogs(self.module.logger,level='WARNING'):
            self.assertIsNone(self.module.Evaluate('key','Q','A').auto_evaluate())

    def test_invalid_scores_rejected(self):
        self.scores['relevance']=10
        self.create.return_value.choices[0].message.content=json.dumps(self.scores)
        with self.assertLogs(self.module.logger,level='WARNING'):
            self.assertIsNone(self.module.Evaluate('key','Q','A').auto_evaluate())

    def test_truncation_disclosed(self):
        payload = self.module.Evaluate('key','Q','A',{'memories':'x'*9000})._payload()
        self.assertIn('memories',payload['truncated_fields'])
        self.assertTrue(payload['memories'].endswith('[TRUNCATED]'))

    def test_log_failure_does_not_discard_scores(self):
        self.module.LOG_PATH.write_text('invalid old log')
        with self.assertLogs(self.module.logger,level='WARNING'):
            self.assertEqual(self.module.Evaluate('key','Q','A').auto_evaluate(),self.scores)
        self.assertEqual(self.module.LOG_PATH.read_text(),'invalid old log')
