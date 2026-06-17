import os
import json
from groq import Groq

class Evaluate:
    def __init__(self, api_key, user_query, response):
        self.api_key = api_key
        self.user_query = user_query
        self.response = response

    def auto_evaluate(self):
        client = Groq(api_key=self.api_key)

        prompt = f"""You are an AI evaluator. Evaluate this chatbot response.

            User Question : {self.user_query}
            Bot Response  : {self.response}

            Rate on scale of 1-5:
            1. Correctness  : Is the answer factually correct?
            2. Relevance    : Does it directly answer the question?
            3. Completeness : Is the answer complete?

            Respond ONLY in JSON:
            {{"correctness": X, "relevance": X, "completeness": X, "reason": "one line"}}"""

        result = client.chat.completions.create(
            model   = "openai/gpt-oss-120b",
            messages= [{"role": "user", "content": prompt}]
        )

        try:
            scores = json.loads(result.choices[0].message.content)

            # ── Save to log ───────────────────────────────
            log_entry = {
                "query"   : self.user_query,
                "response": self.response[:200],
                "scores"  : scores
            }

            log_path = "Evaluation/eval_log.json"
            os.makedirs("Evaluation", exist_ok=True)

            # Load existing log
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log = json.load(f)
            else:
                log = []

            log.append(log_entry)

            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

            return scores

        except:
            return None