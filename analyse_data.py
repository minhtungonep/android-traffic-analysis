import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def create_visualisation(df, plot_type, filename, **kwargs):
    """
    Generic function to create and save visualisations with error handling
    
    Parameters:
    df: DataFrame to use
    plot_type: Type of plot ('histogram', 'boxplot', 'scatter', 'cdf')
    filename: Filename to save the plot
    **kwargs: Additional arguments specific to the plot type
    """
    try:
        plt.figure(figsize=kwargs.get('figsize', (12, 6)))
        
        if plot_type == 'histogram':
            sns.histplot(data=df, x=kwargs.get('x'), hue=kwargs.get('hue', None), 
                        bins=kwargs.get('bins', 30), kde=kwargs.get('kde', True), 
                        element=kwargs.get('element', 'step'))
            if 'threshold' in kwargs:
                plt.axvline(x=kwargs['threshold'], color='red', linestyle='--', 
                            label=kwargs.get('threshold_label', f"Threshold ({kwargs['threshold']})"))
                plt.legend(title=kwargs.get('legend_title', None))
                
        elif plot_type == 'boxplot':
            sns.boxplot(data=df, x=kwargs.get('x'), y=kwargs.get('y'), 
                       hue=kwargs.get('hue', None), palette=kwargs.get('palette', None))
            
        elif plot_type == 'scatter':
            sns.scatterplot(data=df, x=kwargs.get('x'), y=kwargs.get('y'), 
                           hue=kwargs.get('hue', None), palette=kwargs.get('palette', None), 
                           alpha=kwargs.get('alpha', 0.7))
            
            if all(k in kwargs for k in ['vline', 'hline']):
                plt.axvline(x=kwargs['vline'], color='red', linestyle='--')
                plt.axhline(y=kwargs['hline'], color='red', linestyle='--')
            
            if 'text' in kwargs and 'text_pos' in kwargs:
                plt.text(kwargs['text_pos'][0], kwargs['text_pos'][1], kwargs['text'], 
                        bbox=dict(facecolor='white', alpha=0.8))
                
        elif plot_type == 'cdf':
            for group_name, data in kwargs.get('groups', {}).items():
                sorted_data = np.sort(data)
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                plt.plot(sorted_data, cdf, label=group_name, color=kwargs.get('colors', {}).get(group_name))
            
            if 'xlim' in kwargs:
                plt.xlim(kwargs['xlim'])
            
            if 'median_lines' in kwargs:
                for group, median in kwargs['median_lines'].items():
                    color = kwargs.get('colors', {}).get(group, 'gray')
                    plt.axvline(x=median, color=color, linestyle='--', alpha=0.7)
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        plt.title(kwargs.get('title', f"{plot_type.capitalize()} Plot"))
        plt.xlabel(kwargs.get('xlabel', ''))
        plt.ylabel(kwargs.get('ylabel', ''))
        
        if 'annotations' in kwargs:
            for annotation in kwargs['annotations']:
                plt.annotate(text=annotation['text'], 
                            xy=annotation['xy'], 
                            xytext=annotation['xytext'],
                            arrowprops=annotation.get('arrowprops', dict(arrowstyle="->", color=annotation.get('color', 'black'))))
        
        plt.tight_layout()
        plt.savefig(f"analysis_plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created and saved {plot_type} visualisation to analysis_plots/{filename}")
        
    except Exception as e:
        print(f"Error creating {plot_type} visualisation: {type(e).__name__}: {str(e)}")
        plt.close()  # Ensure any open figure is closed even if there's an error

def calculate_cles_safely(group1, group2, max_items=5000):
    """
    Calculate Common Language Effect Size safely for large datasets
    
    Parameters:
    group1, group2: Data groups to compare
    max_items: Maximum number of items to sample from each group
    
    Returns:
    float: CLES value
    """
    # For very large datasets, use sampling
    if len(group1) * len(group2) > 1000000:  # Arbitrary threshold
        # Use the smaller of max_items or the actual sample size for each group
        sample_size1 = min(max_items, len(group1))
        sample_size2 = min(max_items, len(group2))
        
        print(f"Using sampling for CLES calculation ({sample_size1} x {sample_size2} items)...")
        group1_sample = group1.sample(n=sample_size1, random_state=42)
        group2_sample = group2.sample(n=sample_size2, random_state=42)
        
        # Convert to numpy arrays for faster processing
        arr1 = np.array(group1_sample)
        arr2 = np.array(group2_sample)
        
        # Use broadcasting for efficient calculation
        comparisons = arr1[:, np.newaxis] > arr2
        cles = np.mean(comparisons)
        return cles, True  # Return the value and a flag indicating sampling was used
    else:
        # For smaller datasets, use full calculation
        # Convert to numpy arrays for faster processing
        arr1 = np.array(group1)
        arr2 = np.array(group2)
        
        # Use broadcasting for more efficient calculation
        comparisons = arr1[:, np.newaxis] > arr2
        cles = np.mean(comparisons)
        return cles, False  # Return the value and a flag indicating full calculation was used

def perform_statistical_test(test_type, **kwargs):
    """
    Perform statistical tests with proper error handling
    
    Parameters:
    test_type: Type of statistical test ('binomial', 'mannwhitney', etc.)
    **kwargs: Additional arguments specific to the test
    
    Returns:
    dict: Results of the test
    """
    results = {}
    
    try:
        if test_type == 'binomial':
            from scipy.stats import binomtest
            
            if not all(k in kwargs for k in ['successes', 'trials', 'p']):
                raise ValueError("Binomial test requires 'successes', 'trials', and 'p' parameters")
                
            result = binomtest(kwargs['successes'], kwargs['trials'], 
                              p=kwargs['p'], 
                              alternative=kwargs.get('alternative', 'greater'))
            
            results['p_value'] = result.pvalue
            results['test_statistic'] = result.statistic
            results['success_ratio'] = kwargs['successes'] / kwargs['trials']
            results['hypothesis_threshold'] = kwargs['p']
            results['reject_null'] = result.pvalue < kwargs.get('alpha', 0.05)
            results['test_type'] = 'binomial'
            results['alternative'] = kwargs.get('alternative', 'greater')
            
        elif test_type == 'mannwhitney':
            from scipy.stats import mannwhitneyu
            
            if not all(k in kwargs for k in ['group1', 'group2']):
                raise ValueError("Mann-Whitney U test requires 'group1' and 'group2' parameters")
                
            result = mannwhitneyu(kwargs['group1'], kwargs['group2'], 
                                 alternative=kwargs.get('alternative', 'two-sided'))
            
            results['p_value'] = result.pvalue
            results['test_statistic'] = result.statistic
            results['reject_null'] = result.pvalue < kwargs.get('alpha', 0.05)
            results['test_type'] = 'mannwhitney'
            results['alternative'] = kwargs.get('alternative', 'two-sided')
            
            # Calculate CLES if requested
            if kwargs.get('calculate_cles', True):
                try:
                    cles_value, was_sampled = calculate_cles_safely(
                        kwargs['group1'], kwargs['group2'], 
                        max_items=kwargs.get('max_cles_items', 5000)
                    )
                    results['cles'] = cles_value
                    results['cles_sampled'] = was_sampled
                except Exception as cles_error:
                    print(f"Warning: CLES calculation failed: {str(cles_error)}")
                    results['cles'] = None
                    results['cles_error'] = str(cles_error)
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
    except Exception as e:
        print(f"Error performing {test_type} test: {type(e).__name__}: {str(e)}")
        results['error'] = f"{type(e).__name__}: {str(e)}"
        
    return results

# Load the cleaned dataset
def load_and_analyse_data(file_path):
    """
    Load the cleaned dataset and perform analysis for both hypotheses with robust error handling
    """
    # Add file existence check
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please check the path and try again.")
    
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"The file '{file_path}' exists but is empty (0 bytes).")
    
    print(f"Loading cleaned data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        error_type = type(e).__name__
        raise TypeError(f"Failed to load CSV file: {error_type}: {str(e)}. Please ensure the file is a valid CSV.")
    
    # Validate required columns exist
    required_columns = ['dns_query_times', 'tcp_packets', 'volume_bytes', 'type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {', '.join(missing_columns)}. Cannot proceed with analysis.")
    
    # Validate traffic types
    valid_types = ['benign', 'malicious']
    actual_types = df['type'].unique().tolist()
    invalid_types = [t for t in actual_types if t not in valid_types]
    
    if invalid_types:
        raise ValueError(f"Dataset contains invalid traffic types: {invalid_types}. Only 'benign' and 'malicious' are supported.")
    
    # Basic info about the cleaned dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Traffic type distribution:\n{df['type'].value_counts()}")
    
    # Create a directory for saving plots if it doesn't exist
    if not os.path.exists('analysis_plots'):
        os.makedirs('analysis_plots')
    
    # Analyse both hypotheses with try-except blocks
    hypothesis_one_results = None
    hypothesis_two_results = None
    
    try:
        hypothesis_one_results = analyse_hypothesis_one(df)
        print("Hypothesis one analysis completed successfully.")
    except Exception as e:
        print(f"Error analyzing hypothesis one: {type(e).__name__}: {str(e)}")
        print("Attempting to continue with hypothesis two...")
    
    try:
        hypothesis_two_results = analyse_hypothesis_two(df)
        print("Hypothesis two analysis completed successfully.")
    except Exception as e:
        print(f"Error analyzing hypothesis two: {type(e).__name__}: {str(e)}")
    
    if hypothesis_one_results is None and hypothesis_two_results is None:
        print("WARNING: Both hypothesis analyses failed. No results were produced.")
    
    return df, hypothesis_one_results, hypothesis_two_results

def analyse_hypothesis_one(df):
    """
    Hypothesis 1: The probability of having benign traffic given DNS query times > 5 
    AND TCP packets > 40 is at least 9%.
    """
    print("\n--- Hypothesis 1 Analysis ---")
    
    # Calculate the conditional probability
    condition_data = df[(df['dns_query_times'] > 5) & (df['tcp_packets'] > 40)]
    condition_benign = condition_data[condition_data['type'] == 'benign']
    
    total_meeting_condition = len(condition_data)
    benign_meeting_condition = len(condition_benign)
    
    if total_meeting_condition == 0:
        raise ValueError("No records meet the specified conditions (DNS > 5 AND TCP > 40)")
    
    conditional_probability = benign_meeting_condition / total_meeting_condition * 100
    
    print(f"Total records meeting both conditions: {total_meeting_condition}")
    print(f"Benign records meeting both conditions: {benign_meeting_condition}")
    print(f"Conditional probability P(benign | DNS > 5 ∩ TCP > 40): {conditional_probability:.2f}%")
    print(f"Hypothesis threshold: 9%")
    
    # Create visualisations for Hypothesis 1 using the generalised function
    
    # 1. Distribution of DNS query times
    try:
        create_visualisation(
            df=df, 
            plot_type='histogram',
            filename='dns_query_distribution.png',
            x='dns_query_times',
            hue='type',
            bins=30,
            kde=True,
            element='step',
            threshold=5,
            threshold_label='Threshold (DNS > 5)',
            title='Distribution of DNS Query Times by Traffic Type',
            xlabel='DNS Query Times',
            ylabel='Count',
            legend_title='Traffic Type'
        )
    except Exception as e:
        print(f"Error creating DNS query distribution plot: {type(e).__name__}: {str(e)}")
    
    # 2. Distribution of TCP packets
    try:
        create_visualisation(
            df=df, 
            plot_type='histogram',
            filename='tcp_packets_distribution.png',
            x='tcp_packets',
            hue='type',
            bins=30,
            kde=True,
            element='step',
            threshold=40,
            threshold_label='Threshold (TCP > 40)',
            title='Distribution of TCP Packets by Traffic Type',
            xlabel='TCP Packets',
            ylabel='Count',
            legend_title='Traffic Type'
        )
    except Exception as e:
        print(f"Error creating TCP packets distribution plot: {type(e).__name__}: {str(e)}")
    
    # 3. Scatter plot of DNS query times vs TCP packets
    try:
        # Filter extreme outliers for better visualisation
        plot_df = df[(df['dns_query_times'] <= 50) & (df['tcp_packets'] <= 500)]
        
        # Create a colourmap with clear distinction between classes
        colors = {'benign': 'blue', 'malicious': 'orange'}
        
        text_info = f"DNS > 5, TCP > 40\nBenign: {benign_meeting_condition}\nMalicious: {total_meeting_condition - benign_meeting_condition}\nP(benign|conditions) = {conditional_probability:.1f}%"
        
        create_visualisation(
            df=plot_df, 
            plot_type='scatter',
            filename='dns_tcp_scatter.png',
            x='dns_query_times',
            y='tcp_packets',
            hue='type',
            palette=colors,
            alpha=0.7,
            vline=5,
            hline=40,
            text=text_info,
            text_pos=(30, 450),
            title='DNS Query Times vs TCP Packets by Traffic Type',
            xlabel='DNS Query Times',
            ylabel='TCP Packets'
        )
    except Exception as e:
        print(f"Error creating DNS vs TCP scatter plot: {type(e).__name__}: {str(e)}")
    
    # Perform statistical test for Hypothesis 1 using the generalised test function
    test_results = perform_statistical_test(
        test_type='binomial',
        successes=benign_meeting_condition,
        trials=total_meeting_condition,
        p=0.09,
        alternative='greater',
        alpha=0.05
    )
    
    if 'error' in test_results:
        print(f"Statistical test error: {test_results['error']}")
    else:
        print(f"\nStatistical test for Hypothesis 1:")
        print(f"Binomial test p-value: {test_results['p_value']:.10f}")
        conclusion = "Reject H0 - The probability is significantly greater than 9%" if test_results['reject_null'] else "Fail to reject H0 - Cannot conclude the probability is greater than 9%"
        print(f"Conclusion: {conclusion}")
    
    # Return comprehensive results
    return {
        'conditional_probability': conditional_probability,
        'total_meeting_condition': total_meeting_condition,
        'benign_meeting_condition': benign_meeting_condition,
        'malicious_meeting_condition': total_meeting_condition - benign_meeting_condition,
        'hypothesis_threshold': 9,
        'statistical_test': test_results
    }

def analyse_hypothesis_two(df):
    """
    Hypothesis 2: There is a massive traffic volume bytes difference between benign and malicious traffic types.
    """
    print("\n--- Hypothesis 2 Analysis ---")
    
    # Get volume bytes by traffic type
    benign_volumes = df[df['type'] == 'benign']['volume_bytes']
    malicious_volumes = df[df['type'] == 'malicious']['volume_bytes']
    
    # Validate data
    if len(benign_volumes) == 0 or len(malicious_volumes) == 0:
        raise ValueError("One or both traffic types have no volume data")
    
    # Calculate descriptive statistics
    benign_stats = {
        'count': len(benign_volumes),
        'mean': benign_volumes.mean(),
        'median': benign_volumes.median(),
        'std': benign_volumes.std(),
        'min': benign_volumes.min(),
        'max': benign_volumes.max(),
        'q1': benign_volumes.quantile(0.25),
        'q3': benign_volumes.quantile(0.75)
    }
    
    malicious_stats = {
        'count': len(malicious_volumes),
        'mean': malicious_volumes.mean(),
        'median': malicious_volumes.median(),
        'std': malicious_volumes.std(),
        'min': malicious_volumes.min(),
        'max': malicious_volumes.max(),
        'q1': malicious_volumes.quantile(0.25),
        'q3': malicious_volumes.quantile(0.75)
    }
    
    # Print summary statistics
    print("\nVolume Bytes Statistics:")
    print("\nBenign Traffic:")
    for key, value in benign_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nMalicious Traffic:")
    for key, value in malicious_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Calculate differences
    mean_difference = abs(benign_stats['mean'] - malicious_stats['mean'])
    median_difference = abs(benign_stats['median'] - malicious_stats['median'])
    mean_ratio = benign_stats['mean'] / malicious_stats['mean'] if malicious_stats['mean'] != 0 else float('inf')
    median_ratio = benign_stats['median'] / malicious_stats['median'] if malicious_stats['median'] != 0 else float('inf')
    
    print(f"\nDifferences:")
    print(f"  Mean difference: {mean_difference:.2f} bytes")
    print(f"  Median difference: {median_difference:.2f} bytes")
    print(f"  Mean ratio (benign/malicious): {mean_ratio:.2f}x")
    print(f"  Median ratio (benign/malicious): {median_ratio:.2f}x")
    
    # Create visualisations for Hypothesis 2 using the generalised function
    
    # 1. Box plot of volume bytes by traffic type
    try:
        # Filter extreme outliers for better visualisation
        plot_df = df[df['volume_bytes'] <= 100000].copy()  # Only include values up to 100k for better visualisation
        
        # Calculate how many samples were filtered out
        benign_filtered = len(benign_volumes) - len(plot_df[plot_df['type'] == 'benign'])
        malicious_filtered = len(malicious_volumes) - len(plot_df[plot_df['type'] == 'malicious'])
        
        create_visualisation(
            df=plot_df, 
            plot_type='boxplot',
            filename='volume_bytes_boxplot.png',
            x='type',
            y='volume_bytes',
            hue='type',
            palette={'benign': 'blue', 'malicious': 'orange'},
            title='Distribution of Volume Bytes by Traffic Type\n(Values ≤ 100,000 bytes)',
            xlabel='Traffic Type',
            ylabel='Volume Bytes'
        )
        
        # Add text annotations after the plot (not supported by our generic function directly)
        plt.figure(figsize=(1, 1))  # Create dummy figure
        plt.figtext(0.5, 0.5, f"Benign: mean={benign_stats['mean']:.0f}, median={benign_stats['median']:.0f}, n={benign_stats['count']} ({benign_filtered} outliers not shown)")
        plt.figtext(0.5, 0.3, f"Malicious: mean={malicious_stats['mean']:.0f}, median={malicious_stats['median']:.0f}, n={malicious_stats['count']} ({malicious_filtered} outliers not shown)")
        plt.close()
    except Exception as e:
        print(f"Error creating volume bytes boxplot: {type(e).__name__}: {str(e)}")
    
    # 2. Cumulative distribution function (CDF) plot
    try:
        median_lines = {
            'Benign': benign_stats['median'],
            'Malicious': malicious_stats['median']
        }
        
        create_visualisation(
            df=df,  # Note: df not directly used for CDF
            plot_type='cdf',
            filename='volume_bytes_cdf.png',
            groups={
                'Benign': benign_volumes,
                'Malicious': malicious_volumes
            },
            colors={
                'Benign': 'blue',
                'Malicious': 'orange'
            },
            xlim=(0, 100000),
            median_lines=median_lines,
            title='Cumulative Distribution Function of Volume Bytes',
            xlabel='Volume Bytes',
            ylabel='Cumulative Probability',
            annotations=[
                {
                    'text': f"Benign median: {benign_stats['median']:.0f}",
                    'xy': (benign_stats['median'], 0.5),
                    'xytext': (benign_stats['median'] + 5000, 0.55),
                    'color': 'blue',
                    'arrowprops': dict(arrowstyle="->", color='blue')
                },
                {
                    'text': f"Malicious median: {malicious_stats['median']:.0f}",
                    'xy': (malicious_stats['median'], 0.5),
                    'xytext': (malicious_stats['median'] + 5000, 0.45),
                    'color': 'orange',
                    'arrowprops': dict(arrowstyle="->", color='orange')
                }
            ]
        )
    except Exception as e:
        print(f"Error creating volume bytes CDF plot: {type(e).__name__}: {str(e)}")
    
    # Perform statistical test for Hypothesis 2 using the generalised test function
    test_results = perform_statistical_test(
        test_type='mannwhitney',
        group1=benign_volumes,
        group2=malicious_volumes,
        alternative='two-sided',
        calculate_cles=True,
        max_cles_items=5000,
        alpha=0.05
    )
    
    if 'error' in test_results:
        print(f"Statistical test error: {test_results['error']}")
    else:
        print(f"\nStatistical test for Hypothesis 2:")
        print(f"Mann-Whitney U statistic: {test_results['test_statistic']}")
        print(f"p-value: {test_results['p_value']:.10f}")
        
        # Report CLES if available
        if 'cles' in test_results:
            sampling_note = " (estimated from samples)" if test_results.get('cles_sampled', False) else ""
            print(f"Common Language Effect Size{sampling_note}: {test_results['cles']:.4f}")
            print(f"Interpretation: A randomly selected benign traffic instance has a {test_results['cles']*100:.1f}% probability of having larger volume bytes than a randomly selected malicious instance.")
        
        conclusion = "Reject H0 - There is a significant difference in volume bytes between traffic types" if test_results['reject_null'] else "Fail to reject H0 - Cannot conclude there is a significant difference"
        print(f"Conclusion: {conclusion}")
    
    # Return comprehensive results
    return {
        'benign_stats': benign_stats,
        'malicious_stats': malicious_stats,
        'differences': {
            'mean_difference': mean_difference,
            'median_difference': median_difference,
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio
        },
        'statistical_test': test_results
    }

# Main function
if __name__ == "__main__":
    try:
        # Load the cleaned data and perform analysis
        df, h1_results, h2_results = load_and_analyse_data('data/android_traffic_clean.csv')
        print("\nAnalysis complete. Visualisations saved in 'analysis_plots' directory.")
        
        # Optionally save results to JSON for further reference
        try:
            import json
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, bool):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                elif obj is None:
                    return None
                else:
                    try:
                        # Try standard conversion
                        return str(obj)
                    except:
                        # If all else fails, return a string representation
                        return f"Unserializable({type(obj).__name__})"
            
            results = {
                'hypothesis_one': convert_to_serializable(h1_results),
                'hypothesis_two': convert_to_serializable(h2_results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('analysis_plots/analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2)
                
            print("Analysis results saved to analysis_plots/analysis_results.json")
            
        except Exception as e:
            print(f"Warning: Could not save results to JSON: {type(e).__name__}: {str(e)}")
        
    except Exception as e:
        print(f"Critical error: {type(e).__name__}: {str(e)}")