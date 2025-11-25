# GA-Based E-puck Line Following Project - Detailed Logic Mind Map

## 1. Project Overview
```
GA Training System for E-puck Robot
├── Goal: Train robot to follow line and avoid obstacles in circuit arena
├── Method: Genetic Algorithm (GA) optimization
├── Components:
│   ├── Supervisor (supervisorGA_starter.py) - GA controller
│   ├── Robot Controller (epuck_starter.py) - Robot behavior
│   ├── GA Module (ga.py) - Genetic operations
│   └── MLP Network (mlp.py) - Neural network
```

---

## 2. System Architecture

### 2.1 Communication Flow
```
Supervisor (GA Controller)
    ↓ (Emitter sends genotype)
    ├─→ Robot Controller receives weights
    │       ↓
    │   Robot executes behavior
    │       ↓
    │   Calculate fitness
    │       ↓
    └─→ Robot sends fitness back (Receiver)
    ↓
Supervisor evaluates population
    ↓
Generate next generation
    ↓
Repeat for N generations
```

### 2.2 Data Flow Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR MODULE                         │
├─────────────────────────────────────────────────────────────┤
│  1. Initialize Population (random weights)                   │
│  2. For each generation:                                     │
│     ├─→ For each individual:                                │
│     │   ├─→ Send genotype → EMITTER                         │
│     │   ├─→ Reset robot position                            │
│     │   ├─→ Run simulation (300s)                           │
│     │   └─→ Receive fitness ← RECEIVER                      │
│     ├─→ Rank population by fitness                          │
│     ├─→ Save best individual                                │
│     └─→ Generate new population (GA operations)             │
└─────────────────────────────────────────────────────────────┘
                            ↕ (Emitter/Receiver)
┌─────────────────────────────────────────────────────────────┐
│                   ROBOT CONTROLLER MODULE                    │
├─────────────────────────────────────────────────────────────┤
│  1. Receive genotype → RECEIVER                             │
│  2. Set MLP weights from genotype                           │
│  3. Main control loop (every 32ms):                         │
│     ├─→ Read sensors                                        │
│     │   ├─→ Ground sensors (3): left, center, right        │
│     │   └─→ Proximity sensors (8): ps0-ps7                 │
│     ├─→ Normalize sensor data                              │
│     ├─→ MLP forward propagation                            │
│     │   └─→ Output: [left_speed, right_speed]              │
│     ├─→ Set motor velocities                               │
│     ├─→ Calculate fitness components                        │
│     │   ├─→ forwardFitness                                 │
│     │   ├─→ followLineFitness                              │
│     │   ├─→ avoidCollisionFitness                          │
│     │   └─→ spinningFitness                                │
│     └─→ Send fitness → EMITTER                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Supervisor Module (supervisorGA_starter.py)

### 3.1 Initialization
```
SupervisorGA.__init__()
├── Simulation Parameters
│   ├── time_step = 32 ms
│   └── time_experiment = 300 s
├── Robot Control
│   ├── robot_node (DEF Controller)
│   ├── trans_field (position)
│   └── rot_field (rotation)
├── Communication
│   ├── emitter (send data to robot)
│   └── receiver (receive data from robot)
├── GA Parameters
│   ├── num_generations = 120
│   ├── num_population = 60
│   └── num_elite = 6
├── Data Storage
│   ├── population[] (current generation)
│   ├── genotypes[] (all evaluated)
│   └── fitness_values[]
└── Display (fitness plot)
```

### 3.2 Main Workflow
```
run_optimization()
├── Step 1: Wait for num_weights from robot
│   └── createRandomPopulation()
│       └── population = random[-1, 1] (60 x num_weights)
│
├── Step 2: For each generation (0 to 119):
│   ├── current_generation = generation
│   ├── For each individual (0 to 59):
│   │   ├── genotype = population[individual]
│   │   ├── evaluate_genotype(genotype, generation)
│   │   │   ├── Send genotype via emitter
│   │   │   ├── Reset robot position
│   │   │   │   ├── INITIAL_TRANS = [0.47, 0.16, 0]
│   │   │   │   └── INITIAL_ROT = [0, 0, 1, 1.57]
│   │   │   ├── run_seconds(300) - Run simulation
│   │   │   │   └── Loop: supervisor.step(32ms)
│   │   │   │       ├── handle_emitter() - Send data
│   │   │   │       │   ├── Send genotype
│   │   │   │       │   ├── Send current_generation
│   │   │   │       │   ├── Send num_generations
│   │   │   │       │   ├── Send real_speed
│   │   │   │       │   └── Send position
│   │   │   │       └── handle_receiver() - Receive fitness
│   │   │   └── Return fitness value
│   │   └── current_population.append((genotype, fitness))
│   │
│   ├── Step 3: Evaluate generation results
│   │   ├── best = getBestGenotype(current_population)
│   │   ├── average = getAverageGenotype(current_population)
│   │   ├── Save best to "Best.npy"
│   │   └── plot_fitness(generation, best, average)
│   │
│   └── Step 4: Generate next generation
│       └── population = population_reproduce(current_population, num_elite)
│
└── Step 5: Optimization complete
```

### 3.3 Communication Handlers
```
handle_receiver()
├── While receiver has messages:
│   ├── receivedData = receiver.getString()
│   ├── Parse message type:
│   │   ├── "weights: X" → num_weights = X
│   │   └── "fitness: X" → receivedFitness = X
│   └── receiver.nextPacket()

handle_emitter()
├── If num_weights > 0:
│   ├── Send "genotype: [weights array]"
│   ├── Send "current_generation: X"
│   ├── Send "num_generations: X"
│   ├── Calculate real_speed from robot velocity
│   ├── Send "real_speed: X"
│   └── Send "position: [x, y, z]"
```

---

## 4. Robot Controller Module (epuck_starter.py)

### 4.1 Initialization
```
Controller.__init__()
├── Robot Parameters
│   ├── time_step = 32 ms
│   └── max_speed = 1 m/s
│
├── MLP Network Architecture
│   ├── number_input_layer = 11 (8 proximity + 3 ground)
│   ├── number_hidden_layer = [12, 10, 8, 6, 4]
│   ├── number_output_layer = 2 (left, right motors)
│   └── Calculate number_weights
│       └── For each layer connection:
│           ├── Input layer: (inputs + 1) * hidden[0]
│           └── Other layers: layer[n-1] * layer[n]
│
├── Hardware Devices
│   ├── Motors
│   │   ├── left_motor (left wheel motor)
│   │   └── right_motor (right wheel motor)
│   ├── Proximity Sensors (8)
│   │   └── ps0 to ps7 (enabled with time_step)
│   ├── Ground Sensors (3)
│   │   ├── gs0 (left_ir)
│   │   ├── gs1 (center_ir)
│   │   └── gs2 (right_ir)
│   └── Communication
│       ├── emitter (send to supervisor)
│       └── receiver (receive from supervisor)
│
└── Tracking Variables
    ├── fitness_values[] (fitness history)
    ├── velocity_left, velocity_right
    ├── real_speed (from supervisor)
    ├── is_on_edge (boundary detection)
    ├── action_number (step counter)
    └── position [x, y, z]
```

### 4.2 Main Control Loop
```
run_robot()
├── While robot.step(time_step) != -1:
│   │
│   ├── Step 1: Communication
│   │   ├── handle_emitter()
│   │   │   ├── Send "weights: num_weights"
│   │   │   └── Send "fitness: current_fitness"
│   │   └── handle_receiver()
│   │       ├── Receive genotype data
│   │       ├── Receive current_generation
│   │       ├── Receive num_generations
│   │       ├── Receive real_speed
│   │       └── Receive position
│   │           └── Check if is_on_edge (|x| > 0.69 or |y| > 0.69)
│   │
│   ├── Step 2: Read Sensors
│   │   ├── Ground Sensors
│   │   │   ├── left = left_ir.getValue()
│   │   │   ├── center = center_ir.getValue()
│   │   │   ├── right = right_ir.getValue()
│   │   │   ├── Clip values [0, 100]
│   │   │   └── Normalize to [0, 1]
│   │   │       └── inputs[0:3] = normalized ground sensors
│   │   │
│   │   └── Proximity Sensors
│   │       ├── For i in range(8):
│   │       │   ├── temp = proximity_sensors[i].getValue()
│   │       │   ├── Clip values [0, 100]
│   │       │   └── Normalize to [0, 1]
│   │       └── inputs[3:11] = normalized proximity sensors
│   │
│   ├── Step 3: Check for New Genotype
│   │   └── check_for_new_genes()
│   │       ├── If new genotype received:
│   │       │   ├── Reshape weights for each layer
│   │       │   ├── Set network.weights = reshaped data
│   │       │   └── Reset fitness_values[]
│   │       └── Else: continue with current weights
│   │
│   ├── Step 4: Neural Network Control
│   │   └── sense_compute_and_actuate()
│   │       ├── output = network.propagate_forward(inputs)
│   │       ├── velocity_left = output[0]
│   │       ├── velocity_right = output[1]
│   │       ├── left_motor.setVelocity(velocity_left * 3)
│   │       └── right_motor.setVelocity(velocity_right * 3)
│   │
│   └── Step 5: Fitness Calculation
│       └── calculate_fitness()
│           ├── Call fitness functions
│           ├── Combine with weights
│           └── Update fitness average
```

### 4.3 Sensor Configuration
```
Sensor Layout
├── Ground Sensors (IR - detect line)
│   ├── gs0 (left_ir)   - Left ground sensor
│   ├── gs1 (center_ir) - Center ground sensor
│   └── gs2 (right_ir)  - Right ground sensor
│   └── Values: < 500 = on line (dark), > 500 = off line (white)
│
└── Proximity Sensors (IR - detect obstacles)
    ├── ps0 - Right front (45°)
    ├── ps1 - Right front (10°)
    ├── ps2 - Right side (90°)
    ├── ps3 - Right rear side (150°)
    ├── ps4 - Rear (180°)
    ├── ps5 - Rear (180°)
    ├── ps6 - Left rear side (210°)
    └── ps7 - Left front (350°)
    └── Values: Higher = closer obstacle (max 4096)
```

---

## 5. Fitness Functions (Core Logic)

### 5.1 Fitness Calculation Flow
```
calculate_fitness()
├── Call individual fitness functions:
│   ├── forwardFitness() → [0, 1]
│   ├── followLineFitness() → [0, 1]
│   ├── avoidCollisionFitness() → [0, 1]
│   └── spinningFitness() → [0, 1]
│
├── Dynamic Weight Mapping (based on generation):
│   ├── Early Stage (0-30% generations):
│   │   └── {forward: 0.55, line: 0.2, collision: 0.2, spin: 0.05}
│   │       └── Focus: Learn basic movement
│   │
│   ├── Middle Stage (30-70% generations):
│   │   └── {forward: 0.25, line: 0.4, collision: 0.2, spin: 0.05}
│   │       └── Focus: Learn line following
│   │
│   └── Late Stage (70-100% generations):
│       └── {forward: 0.20, line: 0.4, collision: 0.35, spin: 0.05}
│           └── Focus: Master obstacle avoidance
│
├── combinedFitness = Σ(fitness_i × weight_i)
├── fitness_values.append(combinedFitness)
└── fitness = mean(fitness_values)
```

### 5.2 Forward Fitness Function
```
forwardFitness()
├── Purpose: Encourage fast forward movement
│
├── Components:
│   ├── 1. Speed Reward
│   │   └── (|left_speed| + |right_speed|) / (2 × max_speed)
│   │       └── Range: [0, 1], higher = faster
│   │
│   ├── 2. Straightness Reward
│   │   └── 1.0 - |left_speed - right_speed| / max_speed
│   │       └── Range: [0, 1], higher = straighter
│   │
│   └── 3. Direction Penalty
│       └── If left_speed < 0 OR right_speed < 0:
│           └── penalty = 0.5 (discourage backward)
│
├── Formula:
│   └── fitness = speed_reward × straightness_reward - direction_penalty
│
├── Penalties:
│   ├── If real_speed < 0.01: fitness -= 0.1 (stuck)
│   └── If is_on_edge: fitness -= 0.2 (boundary)
│
└── Return: max(0, fitness)
```

### 5.3 Follow Line Fitness Function
```
followLineFitness()
├── Purpose: Encourage staying on the line
│
├── Read Ground Sensors:
│   ├── left_sensor = left_ir.getValue()
│   ├── center_sensor = center_ir.getValue()
│   └── right_sensor = right_ir.getValue()
│
├── Components:
│   │
│   ├── 1. Line Detection Reward (40%)
│   │   ├── If center_sensor < 500: reward = 1.0 (on line)
│   │   ├── Elif left OR right < 500: reward = 0.5 (partial)
│   │   └── Else: reward = 0.0 (lost line)
│   │
│   ├── 2. Correction Reward (30%)
│   │   ├── Line on left (left < 500, right > 500):
│   │   │   └── If right_speed > left_speed: reward = 0.8 (turning left)
│   │   ├── Line on right (right < 500, left > 500):
│   │   │   └── If left_speed > right_speed: reward = 0.8 (turning right)
│   │   └── Line centered (center < 500):
│   │       └── If |left_speed - right_speed| < 0.1: reward = 1.0 (straight)
│   │
│   ├── 3. Speed Reward (30%)
│   │   └── If line_detection > 0.5:
│   │       └── (left_speed + right_speed) / (2 × max_speed)
│   │
│   └── 4. Lost Line Penalty
│       └── If all sensors > 500: penalty = 1.0
│
├── Formula:
│   └── fitness = detection×0.4 + correction×0.3 + speed×0.3 - lost_penalty
│
├── Penalties:
│   ├── If is_on_edge: fitness = 0.0
│   └── If real_speed < 0.01: fitness -= 0.5
│
└── Return: max(0, fitness)
```

### 5.4 Avoid Collision Fitness Function
```
avoidCollisionFitness()
├── Purpose: Detect and avoid obstacles
│
├── Read Proximity Sensors:
│   └── sensor_values[0:7] = ps0 to ps7
│
├── Sensor Weights (importance):
│   ├── ps0, ps1, ps7 (front): 0.2 each
│   ├── ps2, ps6 (sides): 0.1 each
│   └── ps3, ps4, ps5 (rear): 0.05 each
│
├── Components:
│   │
│   ├── 1. Safety Score (60%)
│   │   ├── max_front = max(ps0, ps1, ps7)
│   │   ├── If max_front > danger_threshold × 3:
│   │   │   └── return 0.0 (collision!)
│   │   ├── If max_front < danger_threshold:
│   │   │   └── safety = 1.0 (safe)
│   │   └── Else:
│   │       └── safety = 1.0 - (max_front - threshold) / (threshold × 2)
│   │
│   ├── 2. Avoidance Score (40%)
│   │   ├── Detect obstacle side:
│   │   │   ├── left_obstacle = ps7 > threshold
│   │   │   └── right_obstacle = ps0 > threshold
│   │   │
│   │   ├── Response evaluation:
│   │   │   ├── If left_obstacle:
│   │   │   │   └── Should turn right (right_speed < left_speed)
│   │   │   │       ├── Speed diff > 1.5: score = 1.0
│   │   │   │       ├── Speed diff > 1.0: score = 0.7
│   │   │   │       ├── Speed diff > 0.8: score = 0.6
│   │   │   │       └── ... (gradual scoring)
│   │   │   │
│   │   │   └── If right_obstacle:
│   │   │       └── Should turn left (left_speed < right_speed)
│   │   │           └── Similar gradual scoring
│   │   │
│   │   └── Speed modulation:
│   │       └── If obstacle detected:
│   │           └── score × (0.7 + 0.3 × speed_reduction)
│   │
│   └── Formula:
│       └── fitness = safety_score × 0.6 + avoidance_score × 0.4
│
├── Penalties:
│   ├── If is_on_edge: fitness = 0.0
│   ├── If stuck (real_speed < 0.02 but motors active): fitness = 0.0
│   └── If spinning (opposite speeds): fitness = 0.0
│
└── Return: max(0.0, fitness)
```

### 5.5 Spinning Fitness Function
```
spinningFitness()
├── Purpose: Penalize unproductive spinning behavior
│
├── Calculate Angular Velocity:
│   └── angular_velocity = right_speed - left_speed
│       ├── Positive: counterclockwise rotation
│       └── Negative: clockwise rotation
│
├── Components:
│   │
│   ├── 1. In-Place Spinning Detection
│   │   ├── speed_sum = |left_speed| + |right_speed|
│   │   ├── speed_diff = ||left_speed| - |right_speed||
│   │   │
│   │   ├── If speed_sum < 0.1:
│   │   │   └── return 1.0 (stationary, no penalty)
│   │   │
│   │   └── If left_speed × right_speed < 0: (opposite signs)
│   │       ├── similarity = 1.0 - speed_diff / speed_sum
│   │       └── If similarity > 0.8: (spinning!)
│   │           └── return 1.0 - similarity × 0.8
│   │
│   ├── 2. Oscillation Detection
│   │   └── If angular_velocity_history exists:
│   │       ├── Count direction changes in last 10 steps
│   │       └── If changes > 5:
│   │           └── penalty = min(changes / 10.0, 0.6)
│   │               └── return 1.0 - penalty
│   │
│   └── 3. Turn Ratio Evaluation
│       ├── turn_ratio = |angular_velocity| / speed_sum
│       ├── If turn_ratio < 0.3:
│       │   └── return 1.0 (gentle turn, no penalty)
│       └── Else:
│           └── return 1.0 - turn_ratio × 0.2 (moderate penalty)
│
├── Penalties:
│   ├── If real_speed < 0.0001: return 0.0
│   ├── If stuck but motors active: return 0.0
│   └── If spinning (speed_ratio > 0.8 and opposite): return 0.0
│
└── Return: max(0.0, result)
```

---

## 6. Genetic Algorithm Module (ga.py)

### 6.1 Population Reproduction
```
population_reproduce(genotypes, elite)
├── Parameters:
│   ├── genotypes: [(genotype, fitness), ...]
│   ├── elite: number of best individuals to preserve
│   └── crossover_rate: 70%
│
├── Step 1: Rank Population
│   └── rankPopulation(genotypes)
│       └── Sort by fitness (lowest to highest)
│
├── Step 2: Create New Population
│   └── For each individual (backwards, highest to lowest):
│       │
│       ├── If individual is elite:
│       │   └── Clone directly to new_population
│       │
│       ├── Elif random(1,100) > crossover_rate (30% chance):
│       │   └── Clone to new_population (no modification)
│       │
│       └── Else (70% chance):
│           ├── parent1 = selectParent(genotypes)
│           ├── parent2 = selectParent(genotypes)
│           ├── child = crossover(parent1, parent2)
│           ├── offspring = mutation(child)
│           └── new_population.append(offspring)
│
└── Return: new_population
```

### 6.2 Parent Selection (Tournament)
```
selectParent(genotypes)
├── Tournament Selection Method:
│   ├── number_individuals = 5
│   ├── Randomly select 5 individuals from population
│   ├── group = [random individuals]
│   ├── Rank group by fitness
│   └── Return best individual from group
│
└── Ensures: Better individuals more likely to be selected
```

### 6.3 Crossover Operation
```
crossover(parent1, parent2)
├── Method: Single-point crossover at center
│   ├── crossover_point = len(parent1) / 2
│   │
│   ├── For each gene:
│   │   ├── If gene < crossover_point:
│   │   │   └── child[gene] = parent1[gene]
│   │   └── Else:
│   │       └── child[gene] = parent2[gene]
│   │
│   └── Result: child = [parent1_first_half | parent2_second_half]
│
└── Return: child genotype
```

### 6.4 Mutation Operation
```
mutation(child)
├── Parameters:
│   └── mutation_rate = 20%
│
├── For each gene in child:
│   ├── If random(1,100) < mutation_rate (20% chance):
│   │   ├── random_value = uniform(-1.0, 1.0)
│   │   ├── temp = gene + random_value
│   │   ├── Clip temp to [-1, 1]
│   │   └── after_mutation.append(temp)
│   └── Else (80% chance):
│       └── after_mutation.append(gene) (no change)
│
└── Return: mutated genotype
```

### 6.5 Helper Functions
```
getBestGenotype(genotypes)
└── Return: Individual with highest fitness

getAverageGenotype(genotypes)
└── Return: Mean fitness of population

rankPopulation(genotypes)
└── Sort genotypes by fitness (ascending)
```

---

## 7. MLP Network (mlp.py)

### 7.1 Network Structure
```
MLP Neural Network
├── Architecture: [11, 12, 10, 8, 6, 4, 2]
│   ├── Input Layer: 11 neurons
│   │   ├── 3 ground sensors
│   │   └── 8 proximity sensors
│   │
│   ├── Hidden Layers: 5 layers
│   │   ├── Layer 1: 12 neurons
│   │   ├── Layer 2: 10 neurons
│   │   ├── Layer 3: 8 neurons
│   │   ├── Layer 4: 6 neurons
│   │   └── Layer 5: 4 neurons
│   │
│   └── Output Layer: 2 neurons
│       ├── Neuron 0: left wheel velocity
│       └── Neuron 1: right wheel velocity
│
├── Weight Calculation:
│   ├── Layer 0→1: (11+1) × 12 = 144 weights (with bias)
│   ├── Layer 1→2: 12 × 10 = 120 weights
│   ├── Layer 2→3: 10 × 8 = 80 weights
│   ├── Layer 3→4: 8 × 6 = 48 weights
│   ├── Layer 4→5: 6 × 4 = 24 weights
│   └── Layer 5→6: 4 × 2 = 8 weights
│   └── Total: 424 weights
│
└── Activation Function: Typically tanh or sigmoid
```

### 7.2 Forward Propagation
```
propagate_forward(inputs)
├── Input: [11 sensor values]
│
├── For each layer:
│   ├── If first layer:
│   │   ├── Add bias: inputs_with_bias = [1, input[0], ..., input[10]]
│   │   ├── output = activation(inputs_with_bias × weights[0])
│   │   └── Shape: (12,) = (1,12) × (12,)
│   │
│   └── Else:
│       ├── output = activation(previous_output × weights[layer])
│       └── Shape: (n,) = (m,) × (m,n)
│
└── Return: [left_velocity, right_velocity]
```

---

## 8. Training Process Timeline

### 8.1 Single Individual Evaluation
```
Time: 300 seconds (per individual)
├── Step duration: 32 ms
├── Total steps: 300s / 0.032s = 9,375 steps
│
└── Each step:
    ├── Read sensors (11 values)
    ├── MLP forward pass
    ├── Set motor velocities
    ├── Calculate 4 fitness components
    └── Accumulate fitness values
```

### 8.2 Full Training Timeline
```
Complete Training Run
├── Generations: 120
├── Population per generation: 60
├── Time per individual: 300s
│
├── Total evaluations: 120 × 60 = 7,200
├── Total time: 7,200 × 300s = 2,160,000s
│   └── = 36,000 minutes = 600 hours = 25 days
│
└── Fitness Evolution:
    ├── Early (Gen 0-36): Learn basic movement
    ├── Middle (Gen 36-84): Master line following
    └── Late (Gen 84-120): Perfect obstacle avoidance
```

---

## 9. Key Issues and Solutions

### 9.1 Current Problems
```
Problem 1: Robot skips parts of the line
├── Root Cause:
│   ├── followLineFitness threshold too rigid (500)
│   ├── Lost line penalty too harsh
│   └── No recovery mechanism
│
└── Solution:
    ├── Use dynamic thresholds
    ├── Add gradual correction rewards
    └── Implement line recovery behavior

Problem 2: Obstacle avoidance doesn't return to line
├── Root Cause:
│   ├── avoidCollisionFitness ignores ground sensors
│   ├── No guidance back to line after avoidance
│   └── Discrete scoring creates jerky movements
│
└── Solution:
    ├── Incorporate ground sensors in collision fitness
    ├── Add line-seeking behavior when no obstacle
    └── Use smooth, continuous scoring
```

### 9.2 Recommended Improvements
```
Improvement 1: followLineFitness
├── Current: Binary detection (< 500 or > 500)
└── Improved: Normalized continuous values
    ├── line_norm = max(0, (500 - sensor) / 500)
    ├── Smooth correction rewards
    └── Position-based guidance

Improvement 2: avoidCollisionFitness
├── Current: Only uses proximity sensors
└── Improved: Hybrid approach
    ├── Priority 1: Avoid obstacles (proximity sensors)
    ├── Priority 2: Return to line (ground sensors)
    └── Smooth transitions between behaviors

Improvement 3: Dynamic Fitness Weights
├── Current: 3-stage fixed weights
└── Improved: Continuous adaptation
    ├── Early: Emphasize forward movement
    ├── Middle: Emphasize line following
    └── Late: Emphasize obstacle avoidance + line recovery
```

---

## 10. Data Flow Summary

### 10.1 Complete Cycle
```
[START] Supervisor initializes
    ↓
[GEN 0] Create random population (60 individuals)
    ↓
[IND 0] Send genotype[0] to robot
    ↓
Robot receives weights → Set MLP
    ↓
[300s] Robot runs control loop
    ├── Every 32ms:
    │   ├── Sensors → MLP → Motors
    │   └── Calculate fitness
    └── Average fitness over 9,375 steps
    ↓
Robot sends fitness back to supervisor
    ↓
[IND 1-59] Repeat for all individuals
    ↓
Supervisor ranks population
    ↓
Save best individual
    ↓
GA operations: Selection, Crossover, Mutation
    ↓
[GEN 1-119] Repeat for all generations
    ↓
[END] Best individual saved to Best.npy
```

### 10.2 Message Protocol
```
Supervisor → Robot:
├── "genotype: [w0 w1 w2 ... w423]"
├── "current_generation: X"
├── "num_generations: 120"
├── "real_speed: X.XXX"
└── "position: [x, y, z]"

Robot → Supervisor:
├── "weights: 424"
└── "fitness: X.XXX"
```

---

## 11. Performance Metrics

### 11.1 Fitness Components Range
```
All fitness functions return [0, 1]:
├── forwardFitness: [0, 1]
│   └── 1.0 = Fast, straight, forward
│
├── followLineFitness: [0, 1]
│   └── 1.0 = Centered on line, moving fast
│
├── avoidCollisionFitness: [0, 1]
│   └── 1.0 = Safe distance, proper avoidance
│
└── spinningFitness: [0, 1]
    └── 1.0 = No spinning, smooth movement
```

### 11.2 Combined Fitness
```
combinedFitness = weighted sum
├── Early stage: max ≈ 0.55 + 0.2 + 0.2 + 0.05 = 1.0
├── Middle stage: max ≈ 0.25 + 0.4 + 0.2 + 0.05 = 0.9
└── Late stage: max ≈ 0.20 + 0.4 + 0.35 + 0.05 = 1.0

Typical progression:
├── Gen 0-20: fitness ≈ 0.2-0.4 (random behavior)
├── Gen 20-60: fitness ≈ 0.4-0.6 (learning line following)
├── Gen 60-100: fitness ≈ 0.6-0.8 (mastering avoidance)
└── Gen 100-120: fitness ≈ 0.8-0.95 (optimized behavior)
```

---

## End of Mind Map
