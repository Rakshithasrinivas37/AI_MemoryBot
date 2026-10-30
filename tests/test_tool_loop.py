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


def reply(content=None, calls=None, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls), finish_reason=finish_reason
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
        self.sleep = patch.object(self.module.time, 'sleep').start()
        self.addCleanup(patch.stopall)
        self.quiet = patch('builtins.print')
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def test_user_continue_keeps_exact_ending_after_history_is_cleared(self):
        notice = ("\n\n[This answer is incomplete because generation could not finish. "
                  "Please ask me to continue.]")
        partial = "Earlier explanation. " * 300 + "The final step is to adjust the"
        self.bot.remember_answer("Explain training", partial + notice)
        self.memory.build_messages.return_value = [{"role": "user", "content": "Please continue"}]
        self.create.return_value = reply("The final step is to adjust the weights.")
        self.bot.bot("Please continue")
        instructions = [m['content'] for m in self.create.call_args.kwargs['messages']
                        if isinstance(m, dict) and m.get('role') == 'system']
        prompt = next(m for m in instructions if "bridge the gap" in m)
        import json
        context = json.loads(prompt.split('\n', 1)[1])
        self.assertEqual(context['ending'], partial[-4000:])
        self.assertEqual(context['question'], "Explain training")
        self.assertNotIn("Please ask me to continue", prompt)
        self.assertEqual(self.bot._last_answer_context['question'], "Explain training")

    def test_unrelated_question_does_not_get_continuation_instructions(self):
        self.bot.remember_answer("Earlier", "Earlier answer")
        self.create.return_value = reply("Fresh answer.")
        self.bot.bot("How do I continue training a model?")
        self.assertFalse(self.bot.is_continuation_query("How do I continue training a model?"))
        self.assertNotIn("bridge the gap", str(self.create.call_args.kwargs['messages']))

    def test_continue_without_previous_answer_does_not_invent_context(self):
        self.assertIsNone(self.bot.continuation_instruction("continue"))

    def test_truncated_answer_is_continued_before_storing(self):
        self.create.side_effect = [reply("Training uses back", finish_reason="length"),
                                   reply("propagation to update weights.")]
        answer = self.bot.bot("Explain training")
        self.assertEqual(answer, "Training uses backpropagation to update weights.")
        self.memory.add_messages.assert_called_once_with(("Explain training", answer))
        self.assertNotIn('tools', self.create.call_args.kwargs)
        self.assertNotIn('tool_choice', self.create.call_args.kwargs)

    def test_continuation_is_bounded_and_discloses_incomplete_answer(self):
        self.create.side_effect = [reply("More text. ", finish_reason="length") for _ in range(3)]
        answer = self.bot.bot("Explain training")
        self.assertIn("This answer is incomplete", answer)
        self.assertEqual(self.create.call_count, 3)

    def test_truncated_tool_arguments_are_never_executed(self):
        self.create.return_value = reply(calls=[tool_call()], finish_reason="length")
        self.assertIn("couldn’t complete", self.bot.bot("Latest news"))
        self.search.assert_not_called()

    def test_continuation_failure_preserves_partial_answer(self):
        error = FakeBadRequestError("invalid_request_error")
        self.create.side_effect = [reply("Partial answer", finish_reason="length"), error]
        answer = self.bot.bot("Explain training")
        self.assertTrue(answer.startswith("Partial answer"))
        self.assertIn("This answer is incomplete", answer)

    def test_tool_result_final_answer_is_also_continued(self):
        self.create.side_effect = [reply("Evidence says ", finish_reason="length"), reply("it was released.")]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        answer = self.bot.finish_with_results(client, "Latest news", "Today", [])
        self.assertEqual(answer, "Evidence says it was released.")
        self.memory.add_messages.assert_called_once_with(("Latest news", answer))

    def test_continuation_uses_existing_rate_limit_retries(self):
        error = Exception("Rate limited")
        error.status_code = 429
        self.create.side_effect = [reply("Start ", finish_reason="length"), error, reply("finish.")]
        self.assertEqual(self.bot.bot("Explain training"), "Start finish.")
        self.sleep.assert_called_once_with(30)
        self.assertEqual(self.create.call_count, 3)

    def test_token_limit_summarizes_history_and_retries_same_query(self):
        error = Exception("Limit 8000, Requested 8959")
        error.status_code = 413
        history = [{"role": "system", "content": "Keep the rules."},
                   {"role": "user", "content": "Plan Singapore interviews"},
                   {"role": "assistant", "content": "Interview preparation. " * 200},
                   {"role": "user", "content": "Explain more"}]
        original = [dict(m) for m in history]
        self.memory.build_messages.return_value = history
        answer_calls = []
        def respond(**kwargs):
            if 'tools' not in kwargs:
                return reply("The user is preparing for Singapore interviews.")
            answer_calls.append(kwargs)
            if len(answer_calls) == 1:
                raise error
            return reply("Here is more detail.")
        self.create.side_effect = respond
        self.assertEqual(self.bot.bot("Explain more"), "Here is more detail.")
        retry = answer_calls[1]['messages']
        self.assertEqual(retry[-1], original[-1])
        self.assertEqual(retry[1], original[0])
        self.assertIn("Earlier conversation summary", retry[2]['content'])
        self.assertIn("Singapore interviews", retry[2]['content'])
        self.assertEqual(history[1:1 + len(original)], original)
        summary_calls = [c.kwargs for c in self.create.call_args_list if 'tools' not in c.kwargs]
        import json
        fragments = ''.join(json.loads(c['messages'][1]['content'])['fragment'] for c in summary_calls)
        self.assertEqual(json.loads(fragments), original[1:3])
        self.assertTrue(all(len(json.loads(c['messages'][1]['content'])['fragment'].encode('utf-8')) <= 2000
                            for c in summary_calls))
        self.memory.add_messages.assert_called_once_with(("Explain more", "Here is more detail."))
        self.sleep.assert_not_called()

    def test_summary_failure_preserves_history_and_bounds_retries(self):
        error = Exception("Too large")
        error.status_code = 413
        messages = [{"role": "user", "content": "Earlier"},
                    {"role": "assistant", "content": "Long context " * 100},
                    {"role": "user", "content": "Current query"}]
        from copy import deepcopy
        original = deepcopy(messages)
        self.create.side_effect = [error, error, reply("Recovered answer.")]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        result = self.bot.request_completion(client, messages=messages, tools=[])
        self.assertEqual(result.choices[0].message.content, "Recovered answer.")
        self.assertEqual(messages, original)
        self.assertEqual(self.create.call_args.kwargs['messages'][-1], original[-1])
        self.assertEqual(self.create.call_count, 3)
        self.sleep.assert_not_called()

    def test_summary_keeps_current_tool_exchange(self):
        messages = [{"role": "system", "content": "Rules"},
                    {"role": "user", "content": "Earlier"},
                    {"role": "assistant", "content": "Long earlier answer " * 20},
                    {"role": "user", "content": "Current question"},
                    {"role": "assistant", "tool_calls": [{"id": "abc"}]},
                    {"role": "tool", "tool_call_id": "abc", "content": "Evidence"}]
        tail = messages[3:]
        self.create.return_value = reply("Earlier topic summary.")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.assertTrue(self.bot.summarize_retry_history(client, messages))
        self.assertEqual(messages[2:], tail)
        self.assertEqual(messages[0]['content'], "Rules")

    def test_shrinking_keeps_tool_protocol_and_system_rules(self):
        messages = [
            {"role": "system", "content": "Rules.\nRelevant Memories (facts about the user from past conversations):\n" + "fact " * 1000},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "evidence " * 1000},
            {"role": "user", "content": "Keep my entire question."},
        ]
        self.assertTrue(self.bot.shrink_context(messages))
        self.assertTrue(messages[0]['content'].startswith("Rules.\nRelevant Memories"))
        self.assertEqual(messages[1]['tool_calls'], [{"id": "call_1"}])
        self.assertEqual(messages[2]['tool_call_id'], "call_1")
        self.assertLess(len(messages[2]['content']), 9000)
        self.assertEqual(messages[3]['content'], "Keep my entire question.")

    def test_413_exhaustion_is_bounded(self):
        error = Exception("Request too large")
        error.status_code = 413
        self.create.side_effect = error
        self.assertIn("couldn’t complete", self.bot.bot("Hello"))
        self.assertEqual(self.create.call_count, 3)
        self.sleep.assert_not_called()
        self.memory.add_messages.assert_not_called()

    def test_rate_limit_waits_without_shrinking(self):
        error = Exception("Try again later")
        error.status_code = 429
        self.create.side_effect = [error, reply("Hello there.")]
        self.bot.bot("Hello")
        self.sleep.assert_called_once_with(30)
        self.assertEqual(self.create.call_args_list[0].kwargs, self.create.call_args_list[1].kwargs)

    def test_final_answer_also_retries_oversized_evidence(self):
        error = Exception("Request too large")
        error.status_code = 413
        self.create.side_effect = [error, reply("Supported summary.")]
        results = [{"tool": "web_search", "result": "evidence " * 2000}]
        self.assertEqual(self.bot.finish_with_results(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))), "Question", "Today", results),
                         "Supported summary.")
        first, second = [call.kwargs for call in self.create.call_args_list]
        self.assertLess(len(second['messages'][2]['content']), len(first['messages'][2]['content']))
        self.assertEqual(second['messages'][1]['content'], "Question")
        self.assertNotIn('tools', second)
        self.sleep.assert_not_called()

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
        evidence = self.create.call_args.kwargs['messages'][2]['content']
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
        self.assertIn('AI release', request['messages'][2]['content'])
        self.assertIn('No results found', request['messages'][2]['content'])

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
