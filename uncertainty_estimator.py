import random


class UncertaintyEstimator:
    def __init__(self,
                 llm_manager,
                 generate_prompt,
                 verbal=False,
                 generation_args={}
                 ):
        self.llm_manager = llm_manager
        self.generate_prompt = generate_prompt
        self.verbal = verbal
        self.generation_args = generation_args

    def generate_base_score(self, statement, claim=None, support=False, topic=False):
        #print(f"Processing statement: {statement}")
        if "N/A" in statement:
            #print("N/A in statement")
            return 0.0
        prompt, constraints, formatter = self.generate_prompt(
            statement, claim=claim, verbal=self.verbal, support=support, topic=topic
        )
        
        base_score = formatter(self.llm_manager.chat_completion(
            prompt, print_result=True, trim_response=True, **constraints, **self.generation_args))
        
        if isinstance(base_score, str) and "N/A" in base_score:
            #print(f"N/A in {base_score}")
            return 0.0
        
        #print(f"Base score: {base_score}")
        return base_score

    def __call__(self, statement, claim=None, support=False, topic=False):
        return self.generate_base_score(statement, claim=claim, support=support, topic=topic)
