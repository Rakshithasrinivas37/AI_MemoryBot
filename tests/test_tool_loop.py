"""Tool-loop regression tests; no API keys, network, or ChromaDB required."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class FakeBadRequestError(Exception):
    def __init__(self, code):
        self.body = {"error": {"code": code}}


def reply(content=None, calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls)
    )])


def tool_call(arguments='{"query": "latest AI news"}', name="web_search", identifier="call_1"):
    return SimpleNamespace(id=identifier, function=SimpleNamespace(name=name, arguments=arguments))


class ToolLoopTests(unittest.TestCase):
    def setUp(self):
        self.search = Mock(return_value="Search results")
        self.date = Mock(return_value="October 26, 2026")
        self.create = Mock()
        self.memory = Mock()
        self.memory.build_messages.return_value = [{"role": "user", "content": "Latest AI news?"}]
        self.memory.add_messages.return_value = False
        groq = ModuleType("groq")
        groq.Groq = Mock(return_value=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create))))
        groq.BadRequestError = FakeBadRequestError
        memory = ModuleType("Memory.memory_manager")
        memory.MemoryManager = Mock(return_value=self.memory)
        tools = ModuleType("Tools.tools")
        tools.get_current_date = self.date
        tools.get_tools = lambda: ([], {"web_search": self.search, "get_current_date": self.date})
        with patch.dict(sys.modules, {"groq": groq, "Memory.memory_manager": memory, "Tools.tools": tools}):
            spec = importlib.util.spec_from_file_location(
                "bot_under_test", Path(__file__).resolve().parents[1] / "Chatbot" / "bot.py")
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
            self.bot = self.module.Chatbot("test-key", "test-user")
        self.quiet = patch('builtins.print')
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def test_validation_failure_retries_then_searches(self):
        self.create.side_effect = [FakeBadRequestError("tool_use_failed"),
                                   reply(calls=[tool_call()]), reply("Here is the news.")]
        self.assertEqual(self.bot.bot("Latest AI news?"), "Here is the news.")
        self.search.assert_called_once_with(query="latest AI news as of October 26, 2026")
        self.assertEqual(self.create.call_count, 3)
        self.memory.add_messages.assert_called_once()

    def test_repeated_validation_failure_stops(self):
        self.create.side_effect = FakeBadRequestError("tool_use_failed")
        self.assertIn("couldn’t complete", self.bot.bot("Latest AI news?"))
        self.assertEqual(self.create.call_count, 2)
        self.search.assert_not_called()
        self.memory.add_messages.assert_not_called()

    def test_unrelated_bad_request_is_not_retried(self):
        self.create.side_effect = FakeBadRequestError("invalid_request_error")
        with self.assertRaises(FakeBadRequestError):
            self.bot.bot("Latest AI news?")
        self.assertEqual(self.create.call_count, 1)

    def test_round_limit_disables_tools(self):
        self.create.side_effect = [reply(calls=[tool_call()])] * 3 + [reply("Final answer.")]
        self.assertEqual(self.bot.bot("Latest AI news?"), "Final answer.")
        self.assertEqual(self.search.call_count, 3)
        self.assertNotIn('tools', self.create.call_args.kwargs)
        self.assertNotIn('tool_choice', self.create.call_args.kwargs)

    def test_call_limit_replies_to_every_call(self):
        self.create.side_effect = [reply(calls=[tool_call(identifier=str(i)) for i in range(7)]),
                                   reply("Final answer.")]
        self.bot.bot("Latest AI news?")
        self.assertEqual(self.search.call_count, 5)
        evidence = self.create.call_args.kwargs['messages'][1]['content']
        self.assertEqual(evidence.count('Search results'), 5)
        self.assertNotIn('tools', self.create.call_args.kwargs)
        self.assertNotIn('tool_choice', self.create.call_args.kwargs)

    def test_invalid_arguments_never_execute(self):
        for arguments in ['{"cursor": 5, "id": 0}', '{', '[]', '{"query": ""}',
                          '{"query": 5}', '{"query": "news", "id": 0}']:
            with self.subTest(arguments=arguments):
                self.memory.build_messages.return_value = []
                self.create.side_effect = [reply(calls=[tool_call(arguments)]), reply("Please clarify.")]
                self.bot.bot("Latest AI news?")
        self.search.assert_not_called()

    def test_latest_events_force_dated_search(self):
        self.create.side_effect = [reply(calls=[tool_call()]), reply("Here is the news.")]
        self.bot.bot("What are the latest AI events?")
        self.assertEqual(self.create.call_args_list[0].kwargs['tool_choice'],
                         {"type": "function", "function": {"name": "web_search"}})
        self.assertEqual(self.create.call_args_list[1].kwargs['tool_choice'], 'auto')
        self.assertIn('October 26, 2026', self.search.call_args.kwargs['query'])

    def test_date_is_recomputed_each_turn(self):
        self.date.side_effect = ["October 26, 2026", "October 27, 2026"]
        for day in ["October 26, 2026", "October 27, 2026"]:
            self.memory.build_messages.return_value = []
            self.create.side_effect = [reply(calls=[tool_call()]), reply("Here is the news.")]
            self.bot.bot("Latest AI news?")
            self.assertIn(day, self.search.call_args.kwargs['query'])

    def test_historical_search_not_rewritten(self):
        self.create.side_effect = [reply(calls=[tool_call('{"query": "AI news in 2020"}')]),
                                   reply("Historical events.")]
        self.bot.bot("What was the AI news in 2020?")
        self.search.assert_called_once_with(query="AI news in 2020")
        self.assertEqual(self.create.call_args_list[0].kwargs['tool_choice'], 'auto')

    def test_general_question_does_not_force_search(self):
        self.create.return_value = reply("Python is a programming language.")
        self.bot.bot("What is Python?")
        self.assertEqual(self.create.call_args.kwargs['tool_choice'], 'auto')
        self.search.assert_not_called()

    def test_successful_results_survive_later_search_failure(self):
        self.search.side_effect = ['[{"title":"AI release", "href":"https://example.com/release", "body":"Released in July."}]',
                                   'More evidence', '{"error":"No results found"}']
        self.create.side_effect = [reply(calls=[tool_call()])] * 3 + [reply("A July release was reported; no newer release was verified.")]
        answer = self.bot.bot("Latest advancements in AI")
        self.assertIn('July release', answer)
        request = self.create.call_args.kwargs
        self.assertNotIn('tools', request)
        self.assertIn('AI release', request['messages'][1]['content'])
        self.assertIn('No results found', request['messages'][1]['content'])

    def test_validation_exhaustion_summarizes_existing_results(self):
        self.create.side_effect = [reply(calls=[tool_call()]),
                                   FakeBadRequestError("tool_use_failed"),
                                   FakeBadRequestError("tool_use_failed"), reply("Supported summary.")]
        self.assertEqual(self.bot.bot("Latest AI news?"), 'Supported summary.')
        self.assertNotIn('tools', self.create.call_args.kwargs)
        self.assertEqual(self.create.call_count, 4)

    def test_final_generation_failure_returns_source_snippets(self):
        self.search.return_value = '[{"title":"AI release", "href":"https://example.com/release", "body":"Released in July."}]'
        self.create.side_effect = [reply(calls=[tool_call()])] * 3 + [FakeBadRequestError("tool_use_failed")]
        answer = self.bot.bot("Latest AI news?")
        self.assertIn('https://example.com/release', answer)
        self.assertIn('not been independently verified', answer)
        self.assertEqual(answer.count('https://example.com/release'), 1)
        self.memory.add_messages.assert_not_called()

    def test_evaluation_context_survives_memory_save_and_resets(self):
        self.memory.evaluation_context = {
            "conversation": [{"role": "user", "content": "Earlier question"}],
            "memories": "A retrieved memory", "instructions": "Be clear",
        }
        self.memory.add_messages.side_effect = lambda *_: self.memory.evaluation_context['conversation'].clear()
        self.create.side_effect = [reply(calls=[tool_call()]), reply("Final answer.")]
        self.bot.bot("Latest AI news?")
        context = self.bot.evaluation_context
        self.assertEqual(context['conversation'][0]['content'], 'Earlier question')
        self.assertEqual(context['memories'], 'A retrieved memory')
        self.assertEqual(context['tool_results'][0]['arguments']['query'],
                         'latest AI news as of October 26, 2026')
        self.create.side_effect = None
        self.create.return_value = reply("Hello again.")
        self.bot.bot("Hello")
        self.assertEqual(self.bot.evaluation_context['tool_results'], [])

    def test_empty_final_response_is_handled(self):
        self.create.return_value = reply()
        self.assertIn("couldn't generate", self.bot.bot("Hello"))

    def test_model_ignoring_disabled_tools_stops(self):
        self.create.return_value = reply(calls=[tool_call()])
        self.assertIn("couldn’t produce", self.bot.bot("Latest AI news?"))
        self.assertEqual(self.create.call_count, 4)
        self.assertEqual(self.search.call_count, 3)


if __name__ == '__main__':
    unittest.main()
