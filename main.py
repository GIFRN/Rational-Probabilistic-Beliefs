import argparse
import json
import os
import csv
import random

from collections import defaultdict
#from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, roc_auc_score

import Uncertainpy.src.uncertainpy.gradual as grad
from argument_miner import ArgumentMiner
from uncertainty_estimator import UncertaintyEstimator
from llm_managers import HuggingFaceLlmManager, OpenAiLlmManager
import prompt

OPENAI_API_KEY = <ADD KEY HERE>
ANTHROPIC_API_KEY = <ADD KEY HERE>
os.environ["OPENAI_KEY"] = OPENAI_API_KEY

if __name__ == "__main__":
    # Read claims from CSV
    claims = []
    with open('Datasets/rational_probabilistic_beliefs_dataset.csv') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        for row in reader:
            claims.append({
                'original_claim': row[0],
                'negation_claim': row[1],
                'more_specific_claim': row[2],
                'less_specific_claim': row[3]
            })

   

    baseline_prompt_class = prompt.BaselinePrompts()
    am_prompt_class = prompt.ArgumentMiningPrompts()
    ue_prompt_class = prompt.UncertaintyEvaluatorPrompts()

    # Gather lists of valid prompt method names
    baseline_prompts = [func for func in dir(baseline_prompt_class) if "__" not in func]
    am_prompts = [func for func in dir(am_prompt_class) if "__" not in func]
    ue_prompts = [func for func in dir(ue_prompt_class) if "__" not in func]

    parser = argparse.ArgumentParser()
    # model related args
    parser.add_argument("--model-name", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--save-loc", type=str, default="results/")
    parser.add_argument("--baselines", default=True, action="store_true", help="Run baseline experiment if set.")
    parser.add_argument("--direct", default=False, action="store_true")

    # model parameter args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--quantization", type=str, default="8bit", choices=["4bit", "8bit", "none"]
    )
    # generation related args
    parser.add_argument("--baseline-prompt", type=str, choices=baseline_prompts, default="all")
    parser.add_argument("--am-prompt", type=str, choices=am_prompts + ["all"], default="opro")
    parser.add_argument("--ue-prompt", type=str, choices=ue_prompts + ["all"], default="analyst")
    parser.add_argument("--verbal", default=False, action="store_true")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--breadth", type=int, default=1)
    parser.add_argument("--semantics", type=str, choices=["dfquad", "qe", "eb"], default="dfquad")

    parser.add_argument("--subset-size", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--print-results", default=False, action="store_true", help="Print results instead of saving")
    
    parser.add_argument("--cot-baseline", default=False, action="store_true", help="Use chain-of-thought baseline prompts")
    parser.add_argument("--topk-baseline", default=True, action="store_true", 
                        help="Use top-k baseline approach if set.")

    # New baseline test args
    parser.add_argument("--baseline-test", default=False, action="store_true", 
                       help="Test different baseline prompt variations")
    parser.add_argument("--n-baseline-runs", type=int, default=1,
                       help="Number of runs per prompt variation")

    args = parser.parse_args()

    # Set random seed and subset claims if subset_size is specified
    if args.subset_size is not None:
        random.seed(10)  # Set fixed seed for reproducibility
        claims = random.sample(claims, args.subset_size)

    if "openai" in args.model_name:
        print("Using OpenAI model")
        llm_manager = OpenAiLlmManager(
            model_name=args.model_name,
        )
    else:
        llm_manager = HuggingFaceLlmManager(
            model_name=args.model_name,
            quantization=args.quantization,
        )
    generation_args = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
    }


    ############################################################################
    # BASELINE EVALUATION BLOCK
    # If the user sets --baselines=True, we run a simpler evaluation for each
    # claim variant, calling the 'analyst' prompt with topic=True five times,
    # averaging the result, then saving to CSV.
    ############################################################################
    if args.baselines:
        if args.topk_baseline:
            def get_topk_baseline_score(text, n_runs=1):
                from prompt import UncertaintyEvaluatorPrompts
                prompt_fn = UncertaintyEvaluatorPrompts.analyst
                prompt_str, constraints, formatter = prompt_fn(
                            statement=text,
                            numerical=True,
                            topic=True
                        )
                print("Getting top-k baseline score (numeric tokens only)")
                all_weighted_values = []
                for _ in range(n_runs):

                    
                    topk_outputs = llm_manager.get_topk_tokens(prompt_str, top_k=5, **generation_args)

                    # Filter to numeric tokens
                    numeric_candidates = []
                    for token, prob in topk_outputs:

                        print(f"Raw token: '{token}', Probability: {prob}")

                        # For Hugging Face models, we need to handle different token formats
                        if isinstance(llm_manager, HuggingFaceLlmManager):
                            # Try to extract numbers from the token
                            numbers = ''.join(char for char in token if char.isdigit())
                            if numbers:
                                try:
                                    num = int(numbers)
                                    if 0 <= num <= 100:
                                        numeric_candidates.append((num, prob))
                                except ValueError:
                                    pass
                        else:
                            # OpenAI format handling (existing logic)
                            try:
                                num = int(token.strip())
                                if 0 <= num <= 100:
                                    numeric_candidates.append((num, prob))
                            except ValueError:
                                pass

                    # If no valid numeric tokens, skip or default to zero
                    if not numeric_candidates:
                        all_weighted_values.append(0)
                        continue

                    # Calculate weighted average
                    weighted_val = 0.0
                    total_prob = 0.0
                    for val, prob in numeric_candidates:
                        weighted_val += val * prob
                        total_prob += prob

                    if total_prob > 0:
                        weighted_val /= total_prob
                    
                    prob_val = weighted_val/100
                    all_weighted_values.append(prob_val)

                # Average across all runs
                avg_weighted = sum(all_weighted_values) / len(all_weighted_values)
                return avg_weighted, all_weighted_values
            
            baseline_results = []
            limited_claims = claims[: args.subset_size] if args.subset_size else claims

            for claim_set in limited_claims:
                claim_text_orig = claim_set["original_claim"]
                claim_text_neg = claim_set["negation_claim"]
                claim_text_more = claim_set["more_specific_claim"]
                claim_text_less = claim_set["less_specific_claim"]

                orig_avg, orig_list = get_topk_baseline_score(claim_text_orig)
                neg_avg, neg_list = get_topk_baseline_score(claim_text_neg)
                more_avg, more_list = get_topk_baseline_score(claim_text_more)
                less_avg, less_list = get_topk_baseline_score(claim_text_less)

                row_result = {
                    "original_claim": claim_text_orig,
                    "negation_claim": claim_text_neg,
                    "more_specific_claim": claim_text_more,
                    "less_specific_claim": claim_text_less,
                    "original_baseline": orig_avg,
                    "negation_baseline": neg_avg,
                    "more_specific_baseline": more_avg,
                    "less_specific_baseline": less_avg,
                }

                baseline_results.append(row_result)  

            
            if args.print_results:
                print(f"Original claim: {claim_text_orig}")
                print(f"Average score: {orig_avg:.2f}")
                print(f"Individual scores: {[f'{x:.2f}' for x in orig_list]}\n")

                print(f"Negation claim: {claim_text_neg}")
                print(f"Average score: {neg_avg:.2f}")
                print(f"Individual scores: {[f'{x:.2f}' for x in neg_list]}\n")

                print(f"More specific claim: {claim_text_more}")
                print(f"Average score: {more_avg:.2f}")
                print(f"Individual scores: {[f'{x:.2f}' for x in more_list]}\n")

                print(f"Less specific claim: {claim_text_less}")
                print(f"Average score: {less_avg:.2f}")
                print(f"Individual scores: {[f'{x:.2f}' for x in less_list]}\n")
                print("-" * 80 + "\n")

            else:
                os.makedirs(args.save_loc, exist_ok=True)
            
                output_file = os.path.join(args.save_loc, "all_baselines_analyst_logits.csv")
             
          

                with open(output_file, "w", newline="") as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(
                        [
                            "original_claim",
                            "negation_claim",
                            "more_specific_claim",
                            "less_specific_claim",
                            "original_baseline",
                            "negation_baseline",
                            "more_specific_baseline",
                            "less_specific_baseline",
                        ]
                    )
                    for row in baseline_results:
                        writer.writerow([row[key] for key in row_result.keys()])
            

            

        else:    
            def get_baseline_score(text, n_runs=5):
                """
                Executes the chained or normal baseline prompt for 'n_runs' times,
                returns the average score and the list of all scores obtained.
                """
                from prompt import UncertaintyEvaluatorPrompts
                prompt_fn = UncertaintyEvaluatorPrompts.analyst  # We'll call 'analyst' as before.

                all_scores = []
                for _ in range(n_runs):
                    # Build prompt, constraints, and formatter
                    if args.cot_baseline:
                        # Chain-of-thought style constraints
                        prompt_str, constraints, formatter = prompt_fn(
                            statement=text,
                            topic=True,
                            verbal=False,  # or True, if you want a different formatting
                        )
                        # Add chain-of-thought prefix to the prompt itself
                        prompt_str = "Let's think step by step.\n\n" + prompt_str
                        # Keep the original constraints - we still want to end with Likelihood: X%
                        # This ensures the model provides its reasoning and then gives the final score
                    else:
                        # Normal prompt style (existing code)
                        prompt_str, constraints, formatter = prompt_fn(
                            statement=text,
                            topic=True
                        )

                    # Make the actual call
                    response = llm_manager.chat_completion(
                        message=prompt_str,
                        constraint_prefix=constraints.get("constraint_prefix", None),
                        constraint_options=constraints.get("constraint_options", None),
                        constraint_end_after_options=constraints.get("constraint_end_after_options", True),  # Keep this True
                        trim_response=True,
                        apply_template=False,
                        **generation_args
                    )

                    # For chain of thought, we need to extract the final likelihood from potentially longer text
                    if args.cot_baseline:
                        # Find the last occurrence of "Likelihood:" and extract the percentage
                        likelihood_idx = response.rfind("Likelihood:")
                        if likelihood_idx != -1:
                            # Extract just the likelihood portion
                            response = response[likelihood_idx:]
                    
                    numeric_val = formatter(response)
                    all_scores.append(numeric_val)

                avg_score = sum(all_scores) / len(all_scores)
                return avg_score, all_scores

        
      
            

        # If args.baselines was set but NOT topk_baseline, run the existing baseline
            print("\nRunning baseline experiment with 'analyst' prompt (topic=True)...")
            baseline_results = []
            limited_claims = claims[: args.subset_size] if args.subset_size else claims

            for claim_set in limited_claims:
                claim_text_orig = claim_set["original_claim"]
                claim_text_neg = claim_set["negation_claim"]
                claim_text_more = claim_set["more_specific_claim"]
                claim_text_less = claim_set["less_specific_claim"]

                orig_avg, orig_list = get_baseline_score(claim_text_orig)
                neg_avg, neg_list = get_baseline_score(claim_text_neg)
                more_avg, more_list = get_baseline_score(claim_text_more)
                less_avg, less_list = get_baseline_score(claim_text_less)

                row_result = {
                    "original_claim": claim_text_orig,
                    "negation_claim": claim_text_neg,
                    "more_specific_claim": claim_text_more,
                    "less_specific_claim": claim_text_less,

                    # store just the average as before
                    "original_baseline": orig_avg,
                    "negation_baseline": neg_avg,
                    "more_specific_baseline": more_avg,
                    "less_specific_baseline": less_avg,

                    # new columns storing each set of runs
                    "original_baseline_scores": orig_list,
                    "negation_baseline_scores": neg_list,
                    "more_specific_baseline_scores": more_list,
                    "less_specific_baseline_scores": less_list,
                }
                baseline_results.append(row_result)

            # Save or print baseline results
            if args.print_results:
                print("\nBaseline Results Summary:")
                print("=" * 80)
                for i, row in enumerate(baseline_results):
                    print(f"\nClaim set #{i+1}:")
                    print(f"  Original: {row['original_claim']} => {row['original_baseline']:.4f}")
                    print(f"  Negation: {row['negation_claim']} => {row['negation_baseline']:.4f}")
                    print(f"  More specific: {row['more_specific_claim']} => {row['more_specific_baseline']:.4f}")
                    print(f"  Less specific: {row['less_specific_claim']} => {row['less_specific_baseline']:.4f}")
            else:
                os.makedirs(args.save_loc, exist_ok=True)
                if args.cot_baseline:
                    output_file = os.path.join(args.save_loc, "all_baselines_analyst_topic_cot.csv")
                else:   
                    output_file = os.path.join(args.save_loc, "all_baselines_analyst_topic.csv")
                print(f"\nWriting baseline results to {output_file} ...")

                with open(output_file, "w", newline="") as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(
                        [
                            "original_claim",
                            "negation_claim",
                            "more_specific_claim",
                            "less_specific_claim",
                            "original_baseline",
                            "negation_baseline",
                            "more_specific_baseline",
                            "less_specific_baseline",
                            "original_baseline_scores",
                            "negation_baseline_scores",
                            "more_specific_baseline_scores",
                            "less_specific_baseline_scores",
                        ]
                    )
                    for row in baseline_results:
                        writer.writerow(
                            [
                                row["original_claim"],
                                row["negation_claim"],
                                row["more_specific_claim"],
                                row["less_specific_claim"],
                                row["original_baseline"],
                                row["negation_baseline"],
                                row["more_specific_baseline"],
                                row["less_specific_baseline"],
                                row["original_baseline_scores"],
                                row["negation_baseline_scores"],
                                row["more_specific_baseline_scores"],
                                row["less_specific_baseline_scores"],
                            ]
                        )

                print(f"Baseline CSV written to {output_file}")

        # We exit so we do NOT run the normal argument-mining logic.
        exit(0)
    ############################################################################
    # END OF BASELINE EVALUATION BLOCK
    ############################################################################

    # If we are NOT doing the baselines, proceed with the normal pipeline below:

    # Initialize semantics
    if args.semantics == "qe":
        agg_f = grad.semantics.modular.SumAggregation()
        inf_f = grad.semantics.modular.QuadraticMaximumInfluence(conservativeness=1)
    elif args.semantics == "dfquad":
        agg_f = grad.semantics.modular.ProductAggregation()
        inf_f = grad.semantics.modular.LinearInfluence(conservativeness=1)
    elif args.semantics == "eb":
        agg_f = grad.semantics.modular.SumAggregation()
        inf_f = grad.semantics.modular.EulerBasedInfluence()

    # If "all" is chosen, we run all the AM / UE prompts; otherwise just the chosen ones
    if args.am_prompt != "all":
        am_prompts = [args.am_prompt]
    if args.ue_prompt != "all":
        ue_prompts = [args.ue_prompt]

    # Process each claim set
    for am_prompt in am_prompts:
        for ue_prompt in ue_prompts:
            print(f"\nProcessing with AM: {am_prompt}, UE: {ue_prompt}")

            generate_prompt_am = getattr(am_prompt_class, am_prompt)
            generate_prompt_ue = getattr(ue_prompt_class, ue_prompt)

            ue = UncertaintyEstimator(
                llm_manager=llm_manager,
                generate_prompt=generate_prompt_ue,
                verbal=args.verbal,
                generation_args=generation_args,
            )
            am = ArgumentMiner(
                llm_manager=llm_manager,
                generate_prompt=generate_prompt_am,
                depth=args.depth,
                breadth=args.breadth,
                generation_args=generation_args,
            )

            results = []
            claim_probabilities = []

            # Possibly limit the subset
            limited_claims = claims[: args.subset_size] if args.subset_size else claims

            for claim_set in limited_claims:
                claim_probs = {
                    'original_prob': None,
                    'negation_prob': None,
                    'more_specific_prob': None,
                    'less_specific_prob': None
                }
                
                claim_results = {}
                # Process each type of claim
                for claim_type in ['original', 'negation', 'more_specific', 'less_specific']:
                    key = f'{claim_type}_claim'
                    claim_text = claim_set[key]
                    print(f"\nProcessing {claim_type}: {claim_text}")
                    
                    # Get probability using existing pipeline
                    t_base, t_estimated = am.generate_arguments(claim_text, ue)
                    grad.algorithms.computeStrengthValues(t_base, agg_f, inf_f)
                    grad.algorithms.computeStrengthValues(t_estimated, agg_f, inf_f)
                    
                    # Store probability
                    probability = t_estimated.arguments["db0"].strength
                    prob_key = f"{claim_type}_prob"
                    claim_probs[prob_key] = probability

                    # Store full results
                    claim_results[claim_type] = {
                        "base": {
                            "bag": t_base.to_dict(),
                            "prediction": t_base.arguments["db0"].strength,
                        },
                        "estimated": {
                            "bag": t_estimated.to_dict(),
                            "prediction": t_estimated.arguments["db0"].strength,
                        },
                        "claim": claim_text
                    }
                
                claim_probabilities.append(claim_probs)
                results.append(claim_results)

            # After processing claims, either print or save results
            if args.print_results:
                # Print results in a formatted way
                print("\nResults Summary:")
                print("=" * 80)
                for i, (claim_set, probs) in enumerate(zip(limited_claims, claim_probabilities)):
                    print(f"\nClaim Set {i+1}:")
                    print("-" * 40)
                    for claim_type in ['original', 'negation', 'more_specific', 'less_specific']:
                        key = f'{claim_type}_claim'
                        claim_text = claim_set[key]
                        prob = probs[f'{claim_type}_prob']
                        print(f"{claim_type.replace('_', ' ').title()}:")
                        print(f"  Claim: {claim_text}")
                        print(f"  Probability: {prob:.4f}")
            else:
                # Create save directory if it doesn't exist
                os.makedirs(args.save_loc, exist_ok=True)

                # Save probabilities to CSV
                output_file = os.path.join(
                    args.save_loc, 
                    f'all_claims_with_confidence_AM-{am_prompt}_UE-{ue_prompt}.csv'
                )
                
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header + ['original_prob', 'negation_prob', 
                                              'more_specific_prob', 'less_specific_prob'])
                    
                    for i, row in enumerate(limited_claims):
                        writer.writerow([
                            row['original_claim'],
                            row['negation_claim'],
                            row['more_specific_claim'],
                            row['less_specific_claim'],
                            claim_probabilities[i]['original_prob'],
                            claim_probabilities[i]['negation_prob'],
                            claim_probabilities[i]['more_specific_prob'],
                            claim_probabilities[i]['less_specific_prob']
                        ])

                # Save full results to JSON
                experiment_summary = {
                    "arguments": vars(args),
                    "results": results
                }

                json_output_file = os.path.join(
                    args.save_loc,
                    f"full_results_AM-{am_prompt}_UE-{ue_prompt}.json"
                )

                with open(json_output_file, 'w') as f:
                    json.dump(experiment_summary, f, indent=4)

                print(f"\nResults written to {output_file}")
                print(f"Full results written to {json_output_file}")

  
