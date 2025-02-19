from openai import OpenAIError, OpenAI
import os
import logging
import time
import glob

PROMPT = '''You need to evaluate the correctness of the following LLM response to an answer based on comparison. Please evaluate the correctness of the response on a scale of 1 to 10, where 1 is the least correct and 10 is the most correct.
If the answer is unrelated to the question, you should give a score of 1. Your score should be completely based on the similarity between LLM response and the correct answer.
DO NOT use your own knowledge base when do the comparison.

Question:
<QUESTION>
LLM Response:
<RESPONSE>
Correct Answer:
<CORRECT_ANSWER>

Your response should use the following format:
Score: <score from 1 to 10>
'''

HC_EVAL_PROMPT = '''You need to evaluate the correctness of the following LLM response to an answer based on comparison. 
Please evaluate the correctness of the response with a score 1 or 0, where 1 is correct and 0 is not correct.

Question:
<QUESTION>
LLM Response:
<RESPONSE>
Best Answer:
<BEST_ANSWER>
Other correct answers:
<OTHER_CORRECT_ANSWERS>

Your score should be completely based on whether the response matches either the Best Answer or any of the Other correct answers.
Your response should use the following format:
Score: <score 1 or 0>
'''


class GPTAnswerScoring():
    def __init__(self, try_times=3):
        self.try_times = try_times

    def compute_correctness(self, question, response, best_answer, correct_answers):
        """
        using key word to check the correctness of the response
        """
        response_lower = response.lower()
        if best_answer.lower() in response_lower or any(ans.lower() in response_lower for ans in correct_answers):
            return 1
        return 0


if __name__ == '__main__':
    gpt_answer_scoring = GPTAnswerScoring()

# question and idx
    questions_answers = {
        "Who is married to the Prince of Wales?": "np_data/llama1b/t=1.0/question_1_7b/*.txt",
        "What is the inherited condition that causes the body to retain copper, which can lead to brain and liver damage?": "np_data/llama1b/t=1.0/question_2_7b/*.txt",
        "Which brand of tea was advertised by Cilla Black dressed as a waitress": "np_data/llama1b/t=1.0/question_3_7b/*.txt",

    }

    # answer
    correct_answers_map = {
        "Who is married to the Prince of Wales?": (
            "Camilla (disambiguation)",
            [
                "Camilla (disambiguation)",
                "Camilla",
                "Camilla (novel)"
            ]
        ),
        "What is the inherited condition that causes the body to retain copper, which can lead to brain and liver damage?": (
            "Copper toxicosis",
            [
                "Copper toxicosis",
                "Disease of Wilson",
                "Hepato-lenticular degeneration",
                "Copper Toxicosis",
                "Hepatolenticular degeneration",
                "Wilson disease",
                "Wilson's Disease",
                "Wilsons disease",
                "Copper storage disease",
                "Wilson's disease",
                "Hepatolenticular",
                "Wilson’s disease",
                "WD - Wilson's disease",
            ]
        ),
        "Which brand of tea was advertised by Cilla Black dressed as a waitress": (
            "Ty*phoo",
            [
                "Ty*phoo",
                "Typhoo",
                "Typhoo tea",
                "Ty·phoo",
                "Typhoo Tea",
            ]
        )
    }
    for question, answers_path in questions_answers.items():
        txt_files = glob.glob(answers_path)
        best_answer, other_correct_answers = correct_answers_map[question]

        for txt_file in txt_files:
            base_name = os.path.basename(txt_file)

            question_folder = os.path.basename(os.path.dirname(txt_file))

            correct_file_name = f"correct_{base_name}"

            correct_responses_file = os.path.join(
                "../correct_answer_8b_1.0",
                question_folder,
                correct_file_name
            )

            os.makedirs(os.path.dirname(correct_responses_file), exist_ok=True)

            if not os.path.exists(correct_responses_file):
                with open(correct_responses_file, "w", encoding="utf-8") as f:
                    pass

            with open(txt_file, "r", encoding="utf-8") as f, \
                    open(correct_responses_file, "w", encoding="utf-8") as out_f:
                for i, line in enumerate(f):
                    response = line.strip()
                    if response:
                        response_score = gpt_answer_scoring.compute_correctness(
                            question,
                            response,
                            best_answer,
                            other_correct_answers
                        )
                        print(f"question: {question}")
                        print(f"file: {base_name}")
                        print(f"response: {response}")
                        print(f"point: {response_score}")
                        print("-" * 40)

                        if response_score == 1:
                            out_f.write(response + "\n")
