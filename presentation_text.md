# E-Puck Robot Training with Genetic Algorithm
## Presentation Script (3-4 minutes)

---

## Part 1: Training Logic and Architecture (1.5-2 minutes)

Good afternoon everyone. Today I'll present our implementation of evolutionary robotics using Genetic Algorithms to train an e-puck robot in the Webots simulation environment.

**System Architecture:**
Our system consists of two main components working in tandem. First, the Supervisor Controller orchestrates the entire evolutionary process, managing 80 generations with a population of 60 individuals per generation. Second, the E-puck Robot Controller uses a Multi-Layer Perceptron neural network to control the robot's behavior based on evolved weights.

**Neural Network Design:**
The robot's brain is a feedforward neural network with 11 input neurons receiving data from 8 proximity sensors and 3 ground sensors. This information flows through two hidden layers with 8 and 6 neurons respectively, using hyperbolic tangent activation functions. Finally, two output neurons determine the left and right wheel velocities, enabling the robot to navigate autonomously.

**Genetic Algorithm Process:**
The training follows a classic evolutionary cycle. In each generation, the supervisor evaluates all 60 individuals by sending their unique genotypes—156 neural network weights—to the robot controller. Each robot is tested for 150 seconds in the simulation environment. During evaluation, the robot reads its sensors, propagates the values through its neural network, and actuates its motors accordingly. The fitness value is calculated at each timestep and averaged over the entire evaluation period.

**Evolutionary Operators:**
After evaluation, we apply three key genetic operators. First, elitism preserves the top 6 individuals unchanged to the next generation, ensuring we never lose our best solutions. Second, tournament selection chooses parents by randomly selecting 5 individuals and picking the best among them, balancing selection pressure with diversity. Third, we use single-point crossover at the center of the genotype, combining the first half from one parent with the second half from another. Finally, mutation randomly modifies 30% of genes by adding values between -1 and 1, maintaining genetic diversity and enabling exploration of new solutions.

---

## Part 2: Fitness Function Design (1.5-2 minutes)

**Multi-Objective Fitness Strategy:**
The fitness function is the heart of our evolutionary system, guiding the robots toward desired behaviors. We designed a composite fitness function combining four distinct behavioral objectives, each addressing a specific aspect of robot performance.

**Component 1 - Forward Movement Fitness (Initial weight: 50%):**
This component encourages fast, straight-line movement. It rewards high wheel speeds and penalizes speed differences between wheels, promoting forward locomotion. We also penalize backward movement and stuck states where the robot has high motor commands but zero actual velocity, preventing degenerate solutions.

**Component 2 - Line Following Fitness (Evolving weight: 20% → 50%):**
This objective trains the robot to track a line on the ground. It provides maximum reward when the center ground sensor detects the line, partial reward when side sensors detect it, and penalties for losing the line completely. The function also rewards appropriate corrective turning—right turn when the line drifts left, and left turn when it drifts right—ensuring the robot learns proper line-tracking behavior.

**Component 3 - Obstacle Avoidance Fitness (Evolving weight: 25% → 35%):**
Safety is critical in robotics. This component uses weighted proximity sensor readings, with front sensors having the highest importance. It rewards maintaining safe distances from obstacles and penalizes collisions severely. When obstacles are detected, the function rewards appropriate avoidance maneuvers—turning away from the obstacle side—and encourages speed reduction near obstacles.

**Component 4 - Spinning Penalty (Constant weight: 5%):**
This prevents wasteful behaviors. It detects and penalizes in-place rotation where wheels spin in opposite directions with similar speeds, and oscillation where the robot frequently changes turning direction. This ensures the robot develops efficient, purposeful movement patterns.

**Adaptive Weighting Strategy:**
Critically, we employ an adaptive fitness weighting scheme that changes across generations. In early generations (0-30%), we emphasize forward movement at 50% to establish basic locomotion. In middle generations (30-70%), line following increases to 50% as the robot masters tracking. In late generations (70-100%), obstacle avoidance rises to 35% while maintaining line following at 40%, creating a well-balanced, robust behavior.

**Training Results:**
Our training over 80 generations shows clear evolutionary progress. Starting from random weights with a maximum fitness of 0.48, the population rapidly improved, reaching peak performance of 0.67 at generation 14—a 39% improvement. The final generation maintained a fitness of 0.52, demonstrating stable, competent behavior. The evolved robots successfully navigate the environment, follow lines, avoid obstacles, and move efficiently—all emergent behaviors from the evolutionary process guided by our carefully designed fitness function.

Thank you for your attention. I'm happy to answer any questions about the implementation or results.

---

## Key Statistics to Mention:
- **Population Size**: 60 individuals per generation
- **Generations**: 80 total
- **Evaluation Time**: 150 seconds per individual
- **Neural Network**: 11 → 8 → 6 → 2 architecture (156 total weights)
- **Elite Size**: 6 individuals preserved
- **Crossover Rate**: 50%
- **Mutation Rate**: 30%
- **Best Fitness**: 0.6659 (Generation 14)
- **Final Fitness**: 0.5244 (Generation 79)
- **Improvement**: 8.78% from first to last generation

---

## Suggested Talking Points for Q&A:

**Q: Why did fitness decrease after generation 14?**
A: This is common in evolutionary algorithms. The early peak likely represents a local optimum that's easy to reach but not globally optimal. The later stabilization around 0.52 represents a more robust, generalizable solution that balances all four fitness objectives better.

**Q: Why use adaptive fitness weights?**
A: Training complex behaviors all at once is difficult. By focusing on basic movement first, then line following, then obstacle avoidance, we create a curriculum that guides evolution through increasingly complex behaviors—similar to how we teach children.

**Q: How long does training take?**
A: With 80 generations × 60 individuals × 150 seconds = 720,000 simulation seconds, or about 200 hours of simulated time. However, Webots can run faster than real-time, so actual training takes significantly less wall-clock time.

**Q: Could this work on a real robot?**
A: Yes! The beauty of simulation-based evolution is that successful controllers can be transferred to real robots. However, you'd need to account for the reality gap—differences between simulation and real-world physics—possibly through additional fine-tuning or robust fitness functions.
