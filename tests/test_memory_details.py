"""Saved-memory regression tests without live Groq or ChromaDB calls."""
import ast
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryDetailsTests(unittest.TestCase):
    def setUp(self):
        groq = ModuleType('groq')
        self.client = Mock()
        groq.Groq = Mock(return_value=self.client)
        groq.RateLimitError = type('RateLimitError', (Exception,), {})
        vectors = ModuleType('RAG.vector_store')
        self.db = Mock()
        vectors.get_db = Mock(return_value=self.db)
        vectors.retrieve_from_database = Mock(return_value='')
        self.store = vectors.store_in_database = Mock(return_value=True)
        short = load_module('test_short', ROOT/'Memory/short_term.py')
        long = load_module('test_long', ROOT/'Memory/long_term.py')
        with patch.dict(sys.modules, {'groq':groq, 'RAG.vector_store':vectors,
                                     'Memory.short_term':short, 'Memory.long_term':long}):
            self.module = load_module('memory_details_under_test', ROOT/'Memory/memory_manager.py')
        self.module.PAUSE_BEFORE_SUMMARY_SECONDS = 0
        sleeper = patch.object(self.module.time, 'sleep')
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)
        self.manager = self.module.MemoryManager('alice', 'test-key')
        self.client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                'title':'Singapore AI Job Search', 'summary':'Alice discussed AI jobs and interview preparation.'})))])
        self.quiet = patch('builtins.print')
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def add_batch(self):
        return [self.manager.add_messages((f'Question {i}', f'Full answer {i}')) for i in range(5)]

    def test_fifth_exchange_stores_title_summary_and_exact_transcript(self):
        self.assertEqual(self.add_batch(), [False,False,False,False,True])
        self.store.assert_called_once()
        args, kwargs = self.store.call_args
        self.assertEqual(args[1], 'alice')
        self.assertEqual(kwargs['title'], 'Singapore AI Job Search')
        self.assertEqual(kwargs['conversations'], [
            {'user':f'Question {i}', 'assistant':f'Full answer {i}'} for i in range(5)])
        self.assertFalse(self.manager.pending_exchanges)
        self.assertEqual(len(self.manager.short_term_memory), 0)

    def test_failed_save_retains_first_five_when_sixth_arrives(self):
        self.store.return_value = False
        self.add_batch()
        self.assertEqual(len(self.manager.pending_exchanges), 5)
        self.store.return_value = True
        self.assertTrue(self.manager.add_messages(('Sixth question', 'Sixth answer')))
        self.assertEqual(self.store.call_args.kwargs['conversations'][0]['user'], 'Question 0')
        self.assertEqual(len(self.store.call_args.kwargs['conversations']), 5)
        self.assertEqual(self.manager.pending_exchanges, [('Sixth question', 'Sixth answer')])
        self.assertTrue(self.manager.save_on_exit())
        self.assertEqual(self.store.call_args.kwargs['conversations'], [
            {'user':'Sixth question','assistant':'Sixth answer'}])

    def test_malformed_model_output_does_not_lose_conversation(self):
        self.client.chat.completions.create.return_value.choices[0].message.content = '{"title":"Only a title"}'
        self.assertFalse(self.add_batch()[-1])
        self.store.assert_not_called()
        self.assertEqual(len(self.manager.pending_exchanges), 5)

    def test_413_splits_input_and_preserves_transcript(self):
        error = Exception('Request too large')
        error.status_code = 413
        create = self.client.chat.completions.create
        count = [0]
        def respond(**kwargs):
            count[0] += 1
            if count[0] == 1:
                raise error
            return create.return_value
        create.side_effect = respond
        self.manager.pending_exchanges = [('Q', 'Long answer. ' * 200)]
        self.assertTrue(self.manager.summarize_and_store())
        calls = create.call_args_list
        self.assertLess(len(calls[1].kwargs['messages'][1]['content']),
                        len(calls[0].kwargs['messages'][1]['content']))
        self.assertEqual(self.store.call_args.kwargs['conversations'][0]['assistant'], 'Long answer. ' * 200)

    def test_long_unicode_transcript_is_fully_covered(self):
        text = '世界 Python ' * 2000
        chunks = self.manager._summary_chunks(text)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(chunk.encode('utf-8')) <= 4000 for chunk in chunks))
        self.manager.pending_exchanges = [('Question', text)]
        self.assertTrue(self.manager.summarize_and_store())
        calls = self.client.chat.completions.create.call_args_list
        # Last request generates only the overall title from the section titles.
        sent = ''.join(call.kwargs['messages'][1]['content'] for call in calls[:-1])
        self.assertEqual(json.loads(sent), [{'user':'Question','assistant':text}])
        self.assertIn('Section 1:', self.store.call_args.args[0])
        self.assertEqual(self.store.call_args.kwargs['conversations'][0]['assistant'], text)

    def test_partial_section_failure_does_not_save_incomplete_memory(self):
        create = self.client.chat.completions.create
        create.side_effect = [create.return_value, TimeoutError('timeout'),
                              TimeoutError('timeout'), TimeoutError('timeout')]
        original = [('Question', 'long answer ' * 1000)]
        self.manager.pending_exchanges = original.copy()
        self.assertFalse(self.manager.summarize_and_store())
        self.store.assert_not_called()
        self.assertEqual(self.manager.pending_exchanges, original)

    def test_temporary_failure_retries_successfully(self):
        create = self.client.chat.completions.create
        create.side_effect = [TimeoutError('timeout'), create.return_value]
        self.assertTrue(self.add_batch()[-1])
        self.assertEqual(create.call_count, 2)
        self.sleep.assert_called_once_with(2)

    def test_exhausted_413_keeps_unsaved_batch(self):
        error = Exception('Input too large')
        error.status_code = 413
        self.client.chat.completions.create.side_effect = error
        self.assertFalse(self.add_batch()[-1])
        self.assertLessEqual(self.client.chat.completions.create.call_count, 4)
        self.store.assert_not_called()
        self.assertEqual(len(self.manager.pending_exchanges), 5)

    def test_rate_limit_retries(self):
        create = self.client.chat.completions.create
        create.side_effect = [self.module.RateLimitError('rate limited'), create.return_value]
        self.assertTrue(self.add_batch()[-1])
        self.sleep.assert_called_once_with(30)

    def test_saved_and_legacy_memories_are_readable(self):
        self.db.get.return_value = {'ids':['new','old','broken'], 'documents':['New summary','Legacy summary','Broken metadata'],
            'metadatas':[{'title':'A title','conversations_json':json.dumps([{'user':'Q','assistant':'A'}])},
                         None, {'conversations_json':'not JSON'}]}
        records = self.manager.get_all_memories()
        self.assertEqual(records[0]['id'], 'new')
        self.assertEqual(records[0]['title'], 'A title')
        self.assertEqual(records[0]['conversations'], [{'user':'Q','assistant':'A'}])
        self.assertEqual(records[1]['title'], 'Legacy summary')
        self.assertEqual(records[1]['conversations'], [])
        self.assertEqual(records[2]['conversations'], [])

    def test_delete_id_selects_correct_duplicate(self):
        self.db.get.return_value = {'ids':['first','second'], 'documents':['same','same']}
        self.assertEqual(self.manager.delete_memory('same', 'second')['status'], 'success')
        self.db._collection.delete.assert_called_once_with(ids=['second'])
        self.db._collection.delete.reset_mock()
        self.assertEqual(self.manager.delete_memory('same', 'other-user-id')['status'], 'error')
        self.db._collection.delete.assert_not_called()

    def test_storage_uses_scalar_metadata_and_preserves_transcript(self):
        tree = ast.parse((ROOT/'RAG/vector_store.py').read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'store_in_database')
        namespace = {'get_db':Mock(return_value=self.db), 'json':json, 'datetime':datetime, 'uuid':uuid}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<store_in_database>', 'exec'), namespace)
        transcript = [{'user':'Hello\n世界', 'assistant':'A full reply with "quotes"'}]
        self.assertTrue(namespace['store_in_database']('Summary','alice',title='Title',conversations=transcript))
        saved = self.db.add_texts.call_args.kwargs
        self.assertEqual(saved['texts'], ['Summary'])
        self.assertEqual(saved['metadatas'][0]['title'], 'Title')
        self.assertEqual(json.loads(saved['metadatas'][0]['conversations_json']), transcript)
        self.assertTrue(all(isinstance(v, (str,int,float,bool)) for v in saved['metadatas'][0].values()))


if __name__ == '__main__':
    unittest.main()
