import pandas as pd
import sys
import os
from pathlib import Path
import json
import numpy as np
def analyze_confidence_changes(json_file):
    # Read the JSON file
    with open(json_file) as f:
        data = json.load(f)
    
    # Access the results array
    results = []
    for entry in data['results']:  # Add ['results'] to access the correct array
        # Get baseline initial weight

        claim = entry['original']['base']['bag']['arguments']['db0']['argument']
        # Get strengths for each variant
        specific_strength = entry['original']['estimated']['bag']['arguments']['db0']['strength']
        negated_strength = entry['negation']['estimated']['bag']['arguments']['db0']['strength']
        more_specific_strength = entry['more_specific']['estimated']['bag']['arguments']['db0']['strength']
        less_specific_strength = entry['less_specific']['estimated']['bag']['arguments']['db0']['strength']
        
        # Get initial weights for each variant
        specific_weight = entry['original']['estimated']['bag']['arguments']['db0']['initial_weight']
        negated_weight = entry['negation']['estimated']['bag']['arguments']['db0']['initial_weight']
        more_specific_weight = entry['more_specific']['estimated']['bag']['arguments']['db0']['initial_weight']
        less_specific_weight = entry['less_specific']['estimated']['bag']['arguments']['db0']['initial_weight']
        
    
        results.append({
            'original_claim': claim,
            'original_strength':specific_strength,
            'original_weight': specific_weight,
            'negated_strength': negated_strength,
            'negated_weight': negated_weight,
            'more_specific_strength': more_specific_strength,
            'more_specific_weight': more_specific_weight,
            'less_specific_strength': less_specific_strength,
            'less_specific_weight': less_specific_weight
        })
    
    df = pd.DataFrame(results)

    # Analyze strength changes
    print("\nStrength Analysis:")
    print(f"Total samples: {len(df)}")
    

    # More Specific vs Original  
    more_specific_higher = sum(df['more_specific_strength'] > df['original_strength'])
    more_specific_equal = sum(df['more_specific_strength'] == df['original_strength'])
    print(f"\nMore Specific vs Original:")
    print(f"Higher: {more_specific_higher}")
    print(f"Equal: {more_specific_equal}")
    print(f"Lower: {len(df) - more_specific_higher - more_specific_equal}")

    # Less Specific vs Original
    less_specific_higher = sum(df['less_specific_strength'] > df['original_strength'])
    less_specific_equal = sum(df['less_specific_strength'] == df['original_strength'])
    print(f"\nLess Specific vs Original:")
    print(f"Higher: {less_specific_higher}")
    print(f"Equal: {less_specific_equal}")
    print(f"Lower: {len(df) - less_specific_higher - less_specific_equal}")

    # Negation Analysis
    df['negation_deviation'] = abs(1 - (df['original_strength'] + df['negated_strength']))
    deviation_5 = sum(df['negation_deviation'] > 0.05)
    deviation_10 = sum(df['negation_deviation'] > 0.10)
    deviation_15 = sum(df['negation_deviation'] > 0.15)
    
    print(f"\nNegation Analysis:")
    print(f"Samples with deviation > 5%: {deviation_5}")
    print(f"Samples with deviation > 10%: {deviation_10}")
    print(f"Samples with deviation > 15%: {deviation_15}")
    
    print("\nSamples with deviation > 15%:")
    high_deviation = df[df['negation_deviation'] > 0.15]
    # for idx, row in high_deviation.iterrows():
    #     print(f"\nOriginal ({row['original_strength']*100:.1f}%): {row['original_claim']}")
    #     print(f"Deviation: {row['negation_deviation']*100:.1f}%")

    # Initial Weight Analysis
    print("\nInitial Weight Analysis:")
    

    # More Specific vs Original
    more_specific_weight_higher = sum(df['more_specific_weight'] > df['original_weight'])
    more_specific_weight_equal = sum(df['more_specific_weight'] == df['original_weight'])
    print(f"\nMore Specific vs Original Weights:")
    print(f"Higher: {more_specific_weight_higher}")
    print(f"Equal: {more_specific_weight_equal}")
    print(f"Lower: {len(df) - more_specific_weight_higher - more_specific_weight_equal}")

    print("-" * 80)

    # Less Specific vs Original Weights
    less_specific_weight_higher = sum(df['less_specific_weight'] > df['original_weight'])
    less_specific_weight_equal = sum(df['less_specific_weight'] == df['original_weight'])
    print(f"\nLess Specific vs Original Weights:")
    print(f"Higher: {less_specific_weight_higher}")
    print(f"Equal: {less_specific_weight_equal}")
    print(f"Lower: {len(df) - less_specific_weight_higher - less_specific_weight_equal}")

    # Negation Weight Analysis
    df['negation_weight_deviation'] = abs(1 - (df['original_weight'] + df['negated_weight']))
    weight_deviation_5 = sum(df['negation_weight_deviation'] > 0.05)
    weight_deviation_10 = sum(df['negation_weight_deviation'] > 0.10)
    weight_deviation_15 = sum(df['negation_weight_deviation'] > 0.15)
    
    print(f"\nNegation Weight Analysis:")
    print(f"Samples with weight deviation > 5%: {weight_deviation_5}")
    print(f"Samples with weight deviation > 10%: {weight_deviation_10}")
    print(f"Samples with weight deviation > 15%: {weight_deviation_15}")
    
    print("\nSamples with weight deviation > 15%:")
    high_weight_deviation = df[df['negation_weight_deviation'] > 0.15]


def analyze_confidence_changes_csv(csv_file):
    """Analyze confidence changes from a CSV file containing baseline comparisons."""
    df = pd.read_csv(csv_file)
    
    print(f"\nAnalyzing {csv_file}")
    print("-" * 80)

    # Initial Weight Analysis
    print("\nInitial Weight Analysis:")
    
    # More Specific vs Original
    more_specific_weight_higher = sum(df['more_specific_baseline'] > df['original_baseline'])
    more_specific_weight_equal = sum(df['more_specific_baseline'] == df['original_baseline'])
    print(f"\nMore Specific vs Original Weights:")
    print(f"Higher: {more_specific_weight_higher}")
    print(f"Equal: {more_specific_weight_equal}")
    print(f"Lower: {len(df) - more_specific_weight_higher - more_specific_weight_equal}")

    print("-" * 80)

    # Less Specific vs Original Weights
    less_specific_weight_higher = sum(df['less_specific_baseline'] > df['original_baseline'])
    less_specific_weight_equal = sum(df['less_specific_baseline'] == df['original_baseline'])
    print(f"\nLess Specific vs Original Weights:")
    print(f"Higher: {less_specific_weight_higher}")
    print(f"Equal: {less_specific_weight_equal}")
    print(f"Lower: {len(df) - less_specific_weight_higher - less_specific_weight_equal}")

    # Negation Weight Analysis
    df['negation_weight_deviation'] = abs(1 - (df['original_baseline'] + df['negation_baseline']))
    weight_deviation_5 = sum(df['negation_weight_deviation'] > 0.05)
    weight_deviation_10 = sum(df['negation_weight_deviation'] > 0.10)
    weight_deviation_15 = sum(df['negation_weight_deviation'] > 0.15)
    
    print(f"\nNegation Weight Analysis:")
    print(f"Samples with weight deviation > 5%: {weight_deviation_5}")
    print(f"Samples with weight deviation > 10%: {weight_deviation_10}")
    print(f"Samples with weight deviation > 15%: {weight_deviation_15}")
    
    print("\nSamples with weight deviation > 15%:")
    high_weight_deviation = df[df['negation_weight_deviation'] > 0.15]
    # for idx, row in high_weight_deviation.iterrows():
        # print(f"\nOriginal: {row['original_claim']}")
        # print(f"Weight Deviation: {row['negation_weight_deviation']*100:.1f}%")


def compare_multiple_csv_files(directory):
    """Compare analysis across multiple CSV files, grouped by analysis type."""
    
    print("\nComparing Multiple CSV Files")
    print("=" * 80)

    # Add error checking and debugging
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    # Recursively find all CSV files in directory and subdirectories    
    csv_files = []
    for root, dirs, files in os.walk(directory):
        # Skip validation subfolder
        if 'validation' in root:
            continue
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"Error: No CSV files found in '{directory}' or its subdirectories")
        return
        
    print(f"Found {len(csv_files)} CSV files to analyze:")
    for f in csv_files:
        print(f"  - {f}")

    # More Specific Analysis across files
    print("\nMORE SPECIFIC vs ORIGINAL Analysis:")
    print("-" * 80)
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'all_claims' in csv_file:
                more_specific_weight_higher = sum(df['more_specific_prob'] > df['original_prob'])
                more_specific_weight_equal = sum(df['more_specific_prob'] == df['original_prob'])
            else:
                more_specific_weight_higher = sum(df['more_specific_baseline'] > df['original_baseline'])
                more_specific_weight_equal = sum(df['more_specific_baseline'] == df['original_baseline'])
            print(f"\nFile: {csv_file}")
            # print(f"Higher: {more_specific_weight_higher}")
            # print(f"Equal: {more_specific_weight_equal}")
            # print(f"Lower: {len(df) - more_specific_weight_higher - more_specific_weight_equal}")
            print(f"Percentage lower: {(len(df) - more_specific_weight_higher - more_specific_weight_equal)/len(df)*100:.1f}%")
        except pd.errors.ParserError as e:
            print(f"\nError reading file {csv_file}: {str(e)}")
            continue

    # Less Specific Analysis across files  
    print("\nLESS SPECIFIC vs ORIGINAL Analysis:")
    print("-" * 80)
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'all_claims' in csv_file:
                less_specific_weight_higher = sum(df['less_specific_prob'] > df['original_prob'])
                less_specific_weight_equal = sum(df['less_specific_prob'] == df['original_prob'])
            else:
                less_specific_weight_higher = sum(df['less_specific_baseline'] > df['original_baseline'])
                less_specific_weight_equal = sum(df['less_specific_baseline'] == df['original_baseline'])
            print(f"\nFile: {csv_file}")
            # print(f"Higher: {less_specific_weight_higher}")
            # print(f"Equal: {less_specific_weight_equal}")
            # print(f"Lower: {len(df) - less_specific_weight_higher - less_specific_weight_equal}")
            print(f"Percentage higher: {(less_specific_weight_higher)/len(df)*100:.1f}%")
        except pd.errors.ParserError as e:
            print(f"\nError reading file {csv_file}: {str(e)}")
            continue

    # Negation Analysis across files
    print("\nNEGATION Analysis:")
    print("-" * 80)
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'all_claims' in csv_file:
                df['negation_weight_deviation'] = abs(1 - (df['original_prob'] + df['negation_prob']))
            else:
                df['negation_weight_deviation'] = abs(1 - (df['original_baseline'] + df['negation_baseline']))
            weight_deviation_5 = sum(df['negation_weight_deviation'] > 0.05)
            weight_deviation_10 = sum(df['negation_weight_deviation'] > 0.10)
            weight_deviation_15 = sum(df['negation_weight_deviation'] > 0.15)
            percentage_deviation_5 = weight_deviation_5/len(df)*100
            percentage_deviation_10 = weight_deviation_10/len(df)*100
            percentage_deviation_15 = weight_deviation_15/len(df)*100
            
            print(f"\nFile: {csv_file}")
            # print(f"Samples with weight deviation > 5%: {weight_deviation_5}")
            # print(f"Samples with weight deviation > 10%: {weight_deviation_10}")
            # print(f"Samples with weight deviation > 15%: {weight_deviation_15}")
            print(f"Percentage deviation > 5%: {percentage_deviation_5:.1f}%")
            print(f"Percentage deviation > 10%: {percentage_deviation_10:.1f}%")
            print(f"Percentage deviation > 15%: {percentage_deviation_15:.1f}%")


            high_weight_deviation = df[df['negation_weight_deviation'] > 0.15]
        except pd.errors.ParserError as e:
            print(f"\nError reading file {csv_file}: {str(e)}")
            continue
        # if len(high_weight_deviation) > 0:
        #     print("\nSamples with high weight deviation (>15%):")
        #     for idx, row in high_weight_deviation.iterrows():
        #         print(f"\nOriginal: {row['original_claim']}")
        #         print(f"Weight Deviation: {row['negation_weight_deviation']*100:.1f}%")


def analyze_prompt_variations():
    baseline_file = 'results/baseline_test_results.csv'
    print("\nPROMPT VARIATION Analysis:")
    print("-" * 80)
    
    try:
        df = pd.read_csv(baseline_file)
        variations = ['Existing Prompt', 'Variation A', 'Variation B', 'Variation C']
        
        for variation in variations:
            print(f"\n{variation}:")
            print("-" * 40)
            
            # Filter rows for this variation
            var_df = df[df['prompt_variation'] == variation]
            
            # More Specific vs Original Analysis
            more_specific_weight_higher = sum(var_df['more_specific_baseline'] > var_df['original_baseline'])
            more_specific_weight_equal = sum(var_df['more_specific_baseline'] == var_df['original_baseline'])
            print(f"\nMore Specific vs Original Weights:")
            print(f"Higher: {more_specific_weight_higher}")
            print(f"Equal: {more_specific_weight_equal}")
            print(f"Lower: {len(var_df) - more_specific_weight_higher - more_specific_weight_equal}")
            
            # Less Specific vs Original Analysis
            less_specific_weight_higher = sum(var_df['less_specific_baseline'] > var_df['original_baseline'])
            less_specific_weight_equal = sum(var_df['less_specific_baseline'] == var_df['original_baseline'])
            print(f"\nLess Specific vs Original Weights:")
            print(f"Higher: {less_specific_weight_higher}")
            print(f"Equal: {less_specific_weight_equal}")
            print(f"Lower: {len(var_df) - less_specific_weight_higher - less_specific_weight_equal}")
            
            # Negation Weight Analysis
            var_df['negation_weight_deviation'] = abs(1 - (var_df['original_baseline'] + var_df['negation_baseline']))
            weight_deviation_5 = sum(var_df['negation_weight_deviation'] > 0.05)
            weight_deviation_10 = sum(var_df['negation_weight_deviation'] > 0.10)
            weight_deviation_15 = sum(var_df['negation_weight_deviation'] > 0.15)
            
            print(f"\nNegation Weight Analysis:")
            print(f"Samples with weight deviation > 5%: {weight_deviation_5}")
            print(f"Samples with weight deviation > 10%: {weight_deviation_10}")
            print(f"Samples with weight deviation > 15%: {weight_deviation_15}")
            
    except pd.errors.ParserError as e:
        print(f"\nError reading file {baseline_file}: {str(e)}")


if __name__ == "__main__":

    

    compare_multiple_csv_files('results')
