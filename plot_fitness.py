import re
import matplotlib.pyplot as plt
import numpy as np

def parse_fitness_file(filename):
    """
    Parse the fitness.txt file and extract fitness values for each generation
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # Dictionary to store fitness values for each generation
    generation_fitness = {}
    
    # Split by "Generation: X" pattern
    generations = re.split(r'Generation: (\d+)', content)
    
    # Process each generation (skip the first split which is before any generation)
    for i in range(1, len(generations), 2):
        gen_num = int(generations[i])
        gen_data = generations[i + 1]
        
        # Extract all fitness values (format: "individual_number fitness_value")
        fitness_values = []
        lines = gen_data.strip().split('\n')
        
        for line in lines:
            # Match lines with format: "number fitness_value"
            match = re.match(r'^\d+\s+([\d.]+)', line)
            if match:
                fitness = float(match.group(1))
                fitness_values.append(fitness)
        
        if fitness_values:
            generation_fitness[gen_num] = fitness_values
    
    return generation_fitness

def get_max_fitness_per_generation(generation_fitness):
    """
    Extract the maximum fitness value for each generation
    """
    generations = sorted(generation_fitness.keys())
    max_fitness = []
    
    for gen in generations:
        if generation_fitness[gen]:
            max_fit = max(generation_fitness[gen])
            max_fitness.append(max_fit)
        else:
            max_fitness.append(0)
    
    return generations, max_fitness

def plot_max_fitness(generations, max_fitness, save_path='fitness_plot.png'):
    """
    Plot the maximum fitness for each generation and save the image
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the maximum fitness
    plt.plot(generations, max_fitness, 'b-', linewidth=2, marker='o', 
             markersize=4, label='Max Fitness')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Labels and title
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness', fontsize=12, fontweight='bold')
    plt.title('Maximum Fitness per Generation (E-puck GA Training)', 
              fontsize=14, fontweight='bold')
    
    # Add legend
    plt.legend(loc='lower right', fontsize=10)
    
    # Set x-axis to show all generations
    plt.xlim(min(generations), max(generations))
    
    # Add some statistics as text
    avg_fitness = np.mean(max_fitness)
    final_fitness = max_fitness[-1] if max_fitness else 0
    best_fitness = max(max_fitness) if max_fitness else 0
    best_gen = generations[max_fitness.index(best_fitness)] if max_fitness else 0
    
    stats_text = f'Best Fitness: {best_fitness:.4f} (Gen {best_gen})\n'
    stats_text += f'Final Fitness: {final_fitness:.4f}\n'
    stats_text += f'Average Max Fitness: {avg_fitness:.4f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return best_fitness, best_gen, final_fitness, avg_fitness

def main():
    # Parse the fitness file
    print("Parsing fitness.txt...")
    generation_fitness = parse_fitness_file('fitness.txt')
    
    print(f"Found {len(generation_fitness)} generations")
    
    # Get maximum fitness per generation
    generations, max_fitness = get_max_fitness_per_generation(generation_fitness)
    
    print(f"Generations: {min(generations)} to {max(generations)}")
    
    # Print some statistics
    print("\n=== Fitness Statistics ===")
    for i, (gen, fit) in enumerate(zip(generations[:10], max_fitness[:10])):
        print(f"Generation {gen}: Max Fitness = {fit:.6f}")
    print("...")
    for i, (gen, fit) in enumerate(zip(generations[-10:], max_fitness[-10:])):
        print(f"Generation {gen}: Max Fitness = {fit:.6f}")
    
    # Plot and save
    print("\nGenerating plot...")
    best_fitness, best_gen, final_fitness, avg_fitness = plot_max_fitness(
        generations, max_fitness, 'max_fitness_per_generation.png'
    )
    
    print("\n=== Summary Statistics ===")
    print(f"Best Fitness: {best_fitness:.6f} (achieved at Generation {best_gen})")
    print(f"Final Fitness (Gen {generations[-1]}): {final_fitness:.6f}")
    print(f"Average Maximum Fitness: {avg_fitness:.6f}")
    print(f"Improvement: {((final_fitness - max_fitness[0]) / max_fitness[0] * 100):.2f}%")

if __name__ == "__main__":
    main()
