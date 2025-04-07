import re


def baseline_formatter(response):
    if "false" in response.lower() and "true" in response.lower():
        return response.lower().rindex("true") > response.lower().rindex("false")
    elif "true" in response.lower():
        return True
    elif "false" in response.lower():
        return False
    else:
        # Return false as a default
        return False


class BaselinePrompts:
    @staticmethod
    def deniz(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            prompt = f"""
            Claim: {statement}

            Instructions:
                Consider the claim and determine whether it is true or false.
                Think step by step before providing the final answer.
                Utilize critical thinking and logical reasoning in your evaluation.
                Ensure clarity in your reasoning."""

            return prompt, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Answer:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        prompt = f"""
            Claim: {statement}

            Instructions:
                Consider the claim and determine whether it is true or false.
                Utilize critical thinking and logical reasoning in your evaluation.
                Ensure clarity in your reasoning."""

        return prompt, constraints, baseline_formatter

    @staticmethod
    def analyst(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            instructions = (
                f"You are an analyst evaluating whether statements are true or false. "
                f'For the statement:\n\nStatement: "{statement}"\n\nplease determine '
                f"whether the statement is correct, accurate and truthful. "
                "Please think step by step before providing the final answer."
            )

            return instructions, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Result:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        instructions = (
            f"You are an analyst evaluating whether statements are true or false. "
            f'For the statement:\n\nStatement: "{statement}"\n\nplease determine '
            f"whether the statement is correct, accurate and truthful. "
            "Please give the result in the form 'Result: True' if the statement "
            "is true and 'Result: False' if the statement is false."
        )

        return instructions, constraints, baseline_formatter

    @staticmethod
    def gabriel(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            prompt = f"""Please provide an assessment based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, you should consider it false.
    Statement: {statement}
    Now take a deep breath, and determine whether the statement is true or false."""

            return prompt, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Assessment:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        prompt = f"""Please provide an assessment based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, you should consider it false. Please think step by step before providing the final answer.
Statement: {statement}
Now take a deep breath, think step by step and determine whether the statement is true or false."""

        return prompt, constraints, baseline_formatter


class ArgumentMiningPrompts:
   

    @staticmethod
    def gabriel(statement, support=False, **_):
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return argument

        return (
            f"""Please provide a single short argument {"supporting" if support else "attacking"} the following claim. Construct the argument so it refers to the truthfulness of the claim. Only provide an argument if you think there is a valid and convincing {"support" if support else "attack"} for this claim (there is a non-zero probability that the claim is true), otherwise return: N/A.
        Claim: {statement}
        Now take a deep breath and come up with an argument.
        Argument:""",
            {},
            formatter,
        )


class UncertaintyEvaluatorPrompts:
    

    @staticmethod
    def analyst(statement, claim=None, support=False, numerical=False, topic=False, baseline_test=False, **_):
        if not topic and claim is None:
            raise ValueError(
                "Claim is required for the analyst prompt without topic flag, but was None"
            )

        def formatter(output):
            try:
                likelihood = output.replace("Likelihood:", "").replace("%", "").strip()
                likelihood = likelihood.split("\n")[0]
                return int(likelihood) / 100
            except:
                return 0

        constraints = {
            "constraint_prefix": "Likelihood:",
            "constraint_options": [f" {l}%" for l in range(0, 101)],
            "constraint_end_after_options": True,
        }

        if topic and not numerical and baseline_test:
            existing_prompt = (
                "You are an analyst evaluating the validity of statements. "
                f'For the statement:\n\nStatement: "{statement}"\n\nplease give your confidence '
                "that the statement is correct, accurate and truthful. "
                "Your response should be between 0% and 100% with 0% indicating that the "
                "considered statement is definitely invalid, 100% indicating that the considered statement is "
                "definitely valid, and values in between indicating various levels of uncertainty. "
                "Your estimates should be well-calibrated, so feel free to err on the side of caution "
                "and output moderate probabilities if you are not completely sure in your assessment. "
                "Only respond with your confidence score, no other text. "
                "Please respond in the following form:\n\n"
                "Likelihood: The predicted likelihood that the considered statement is valid"
            )

            prompt_a = (
                "Carefully assess the correctness of the following statement. "
                "Focus on facts, not speculation. Provide a single confidence score from 0% (definitely false) "
                "to 100% (definitely true). Use rigorous reasoning but only return the numeric value, "
                "no additional explanation.\n\n"
                f"Statement: \"{statement}\"\n\n"
                "Likelihood: The predicted likelihood that the statement is correct"
            )

            prompt_b = (
                "Assume you have full access to relevant factual sources. Determine how likely "
                "the following statement is true. Answer with a single integer from 0% to 100%. "
                "If uncertain, lean toward moderate values. Do not add any extra text.\n\n"
                f"Statement: \"{statement}\"\n\n"
                "Likelihood: The predicted likelihood that the statement is valid"
            )

            prompt_c = (
                "You specialize in truth-checking. Rate the following statement on a scale of 0% to 100% "
                "based on how factually correct it is, with 0% meaning certainly incorrect and 100% meaning "
                "certainly correct. Output only the integer value and nothing else.\n\n"
                f"Statement: \"{statement}\"\n\n"
                "Likelihood: The predicted likelihood that the statement is valid"
            )

            return [
                ("Existing Prompt", existing_prompt, constraints, formatter),
                ("Variation A", prompt_a, constraints, formatter),
                ("Variation B", prompt_b, constraints, formatter),
                ("Variation C", prompt_c, constraints, formatter),
            ]

        if topic:
            instructions = (
                "You are an analyst evaluating the validity of statements. "
                f'For the statement:\n\nStatement: "{statement}"\n\nplease give your confidence '
                f"that the statement is correct, accurate and truthful. "
                f"Your response should be between 0% and 100% with 0% indicating that the "
                f"considered statement is definitely invalid, 100% indicating that the considered statement is "
            )
            if numerical:
                return ( instructions
            + (
                "definitely valid and values in between indicating various levels of "
                "uncertainty. Your estimates should be well-calibrated, so feel free to "
                "err on the side of caution and output moderate probabilities if you are "
                "not completely sure in your assessment. Only respond with your confidence score, no other text. "
                "Just return a single number between 0 and 100 representing your confidence score, and nothing else." 
            ),
            constraints,
            formatter,
                )
        else:
            instructions = (
                "You are an analyst evaluating the validity and relevance of arguments. "
                f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
                f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
                f'the statement:\n\nStatement: "{claim}"\n\nYour assessment should be based '
                f"on how well the argument {'supports' if support else 'refutes'} the considered "
                "statement as well as the correctness, accuracy and truthfulness of the given argument. "
                f"Your response should be between 0% and 100% with 0% indicating that the "
                f"considered argument is definitely invalid, 100% indicating that the considered argument is "
            )

        return (
            instructions
            + (
                "definitely valid and values in between indicating various levels of "
                "uncertainty. Your estimates should be well-calibrated, so feel free to "
                "err on the side of caution and output moderate probabilities if you are "
                "not completely sure in your assessment. Only respond with your confidence score, no other text."
                "Please respond in the following form:"
                "\n\nLikelihood: The predicted likelihood that the considered "
                f"{'argument' if not topic else 'statement'} is valid"
            ),
            constraints,
            formatter,
        )

    