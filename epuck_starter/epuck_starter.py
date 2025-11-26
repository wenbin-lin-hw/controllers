from controller import Robot, Receiver, Emitter
import sys,struct,math
import numpy as np
import mlp as ntw

class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 1  # m/s
 
        # MLP Parameters and Variables   
        ### Define below the architecture of your MLP network. 
        ### Add the number of neurons for each layer.
        ### The number of neurons should be in between of 1 to 20.
        ### Number of hidden layers should be one or two.
        self.number_input_layer = 11 #8 proximity + 3 ground sensors
        # Example with one hidden layers: self.number_hidden_layer = [5]
        # Example with two hidden layers: self.number_hidden_layer = [7,5]
        self.number_hidden_layer = [12,10,8,6,4]
        self.number_output_layer = 2 
        
        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        
        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []
        
        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1,len(self.number_neuros_per_layer)):
            if(n == 1):
                # Input + bias
                self.number_weights += (self.number_neuros_per_layer[n-1]+1)*self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n]

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0
        self.current_generation = 0
    
        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)
       
        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)
        
        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter") 
        self.receiver = self.robot.getDevice("receiver") 
        self.receiver.enable(self.time_step)
        self.receivedData = "" 
        self.receivedDataPrevious = "" 
        self.flagMessage = False
        # Time tracking
        self.step_count = 0
        self.time_on_line = 0
        self.time_off_line = 0
        self.total_distance = 0
        self.distance_on_line = 0
        # Speed tracking
        self.speed_history = []
        self.avg_speed_on_line = 0
        # Collision tracking
        self.collision_count = 0
        self.near_collision_count = 0
        self.time_near_obstacle = 0
        # Line following tracking
        self.consecutive_on_line = 0
        self.consecutive_off_line = 0
        
        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0
        self.num_generations = 300
        self.real_speed = 0
        self.is_on_edge = False
        self.action_number = 0
        self.position = 0, 0

    def forwardFitness(self):
        """
        Forward movement fitness function

        Parameters:
            left_speed: Left wheel speed
            right_speed: Right wheel speed
            max_speed: Maximum speed
        """
        # Reward mechanism
        # 1. The faster both wheels move, the better (encourage fast movement)
        speed_reward = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)

        # 2. The smaller the speed difference between wheels, the better (encourage straight line movement)
        speed_difference = abs(self.velocity_left - self.velocity_right) / self.max_speed
        straightness_reward = 1.0 - speed_difference

        # 3. Both wheels should rotate forward (penalize backward movement)
        direction_penalty = 0
        if self.velocity_left < 0 or self.velocity_right < 0:
            direction_penalty = 0.5

        # Combined fitness
        fitness = speed_reward * straightness_reward - direction_penalty
        # if self.real_speed <0.005:
        #     fitness = fitness-0.05
        # print("real speed:",self.real_speed)
        # # if self.is_on_edge:
        # #     fitness = 0.0
        # if self.real_speed>=0.07:
        #     fitness = 1.0
        # elif self.real_speed<0.07 and self.real_speed>=0.06:
        #     fitness = 0.8
        # elif self.real_speed<0.06 and self.real_speed>=0.05:
        #     fitness = 0.6
        # elif self.real_speed<0.05 and self.real_speed>=0.04:
        #     fitness = 0.4
        # elif self.real_speed<0.04 and self.real_speed>=0.02:
        #     fitness = 0.2
        # elif self.real_speed<0.02 and
        #     fitness = 0.1
        # else:
        #     fitness = 0.0
        # if self.real_speed < 0.01 and max(abs(self.velocity_left), abs(self.velocity_right)) > 0.5:
        #     return 0.0
        if self.real_speed < 0.01:
            fitness -= 0.1
        if self.is_on_edge:
            # if self.action_number % 100 == 0:
            #     print("is on the edge......")
            fitness -= 0.2
        # if abs(self.velocity_right)!=0
        #     if abs(self.velocity_left)/abs(self.velocity_right)>0.8 and self.velocity_right*self.velocity_left<0:
        #         fitness=0.0

        return max(0, fitness)  # Ensure fitness is non-negative

    def followLineFitness(self):
        """
        Improved line following fitness function

        Key improvements:
        1. Better sensor interpretation with dynamic thresholds
        2. Smooth rewards for gradual corrections
        3. Speed maintenance on the line
        4. Recovery mechanism when line is lost
        """
        left_sensor = self.left_ir.getValue()
        center_sensor = self.center_ir.getValue()
        right_sensor = self.right_ir.getValue()
        left_speed = self.velocity_left
        right_speed = self.velocity_right
        max_speed = self.max_speed

        # Dynamic threshold - adjust based on sensor readings
        # Typically, dark line gives lower values (< 500), white surface gives higher values (> 500)
        line_threshold = 500

        # Calculate how many sensors detect the line
        sensors_on_line = sum([
            1 if left_sensor < line_threshold else 0,
            1 if center_sensor < line_threshold else 0,
            1 if right_sensor < line_threshold else 0
        ])

        # Normalize sensor values for smoother fitness calculation
        left_norm = max(0, (line_threshold - left_sensor) / line_threshold)
        center_norm = max(0, (line_threshold - center_sensor) / line_threshold)
        right_norm = max(0, (line_threshold - right_sensor) / line_threshold)

        # 1. Line detection reward - prioritize center sensor
        line_detection = center_norm * 0.6 + (left_norm + right_norm) * 0.2

        # 2. Direction correction reward - smooth and continuous
        correction_reward = 0.0
        speed_diff = right_speed - left_speed

        # Calculate line position: negative = left, positive = right, 0 = center
        line_position = (right_norm - left_norm)

        # Good behavior: turn towards the line
        if line_position < -0.1 and speed_diff > 0:  # Line on left, turning left
            correction_reward = min(abs(line_position) * abs(speed_diff) / max_speed, 1.0)
        elif line_position > 0.1 and speed_diff < 0:  # Line on right, turning right
            correction_reward = min(abs(line_position) * abs(speed_diff) / max_speed, 1.0)
        elif abs(line_position) < 0.1 and abs(speed_diff) < 0.2 * max_speed:  # Centered, going straight
            correction_reward = 1.0

        # 3. Speed reward - encourage movement when on line
        avg_speed = (abs(left_speed) + abs(right_speed)) / (2 * max_speed)
        if sensors_on_line > 0:
            speed_reward = avg_speed * 0.8
        else:
            # Penalize high speed when line is lost
            speed_reward = -avg_speed * 0.3

        # 4. Stability reward - penalize erratic movements
        stability = 1.0 - min(abs(speed_diff) / (2 * max_speed), 1.0)
        stability_reward = stability * 0.3 if sensors_on_line > 0 else 0

        # 5. Lost line penalty - but allow recovery
        lost_penalty = 0.0
        if sensors_on_line == 0:
            lost_penalty = 0.5  # Moderate penalty to encourage finding the line

        # Combine all components
        fitness = (
                line_detection * 0.35 +
                correction_reward * 0.30 +
                speed_reward * 0.20 +
                stability_reward * 0.15 -
                lost_penalty
        )

        # Edge penalty
        if self.is_on_edge:
            fitness *= 0.1

        # Stuck penalty - robot not moving despite motor commands
        if self.real_speed < 0.01 and max(abs(left_speed), abs(right_speed)) > 0.3:
            fitness *= 0.2

        return max(0.0, min(1.0, fitness))

    # ============================================================================
    # FITNESS FUNCTION 3: AVOID COLLISION FITNESS
    # ============================================================================
    def avoidCollisionFitness(self):
        """
        Improved obstacle avoidance fitness function with line recovery

        Goal: Avoid obstacles while maintaining awareness of the line position
              and actively returning to the line after obstacle avoidance

        Key improvements:
        1. Detect obstacles and avoid them (primary goal)
        2. Monitor line position during avoidance (secondary goal)
        3. Guide robot back to line when obstacle is cleared (recovery behavior)

        Args:
            proximity_sensors: 8 proximity sensor readings [ps0-ps7]
            left_speed: Left wheel speed
            right_speed: Right wheel speed
            danger_threshold: Danger distance threshold

        Returns:
            Fitness score [0, 1]

        Design points:
        1. Danger detection: Identify obstacles in front and sides
        2. Avoidance response: Adjust wheel speeds based on obstacle position
        3. Line awareness: Use ground sensors to stay oriented toward the line
        4. Recovery behavior: After clearing obstacle, turn back toward the line
        """
        sensor_values = []
        for sensor in self.proximity_sensors:
            sensor_values.append(sensor.getValue())
        proximity_sensors = sensor_values
        left_speed, right_speed, danger_threshold = self.velocity_left, self.velocity_right, 90

        # Get ground sensor values for line awareness
        left_ground = self.left_ir.getValue()
        center_ground = self.center_ir.getValue()
        right_ground = self.right_ir.getValue()
        line_threshold = 500

        if len(proximity_sensors) < 8:
            return 0.0

        # Sensor weights (front sensors are most important)
        sensor_weights = np.array([
            0.2,  # ps0 - right front
            0.2,  # ps1 - right front
            0.1,  # ps2 - right side
            0.05,  # ps3 - right rear side
            0.05,  # ps4 - rear
            0.05,  # ps5 - rear
            0.1,  # ps6 - left side
            0.2  # ps7 - left front
        ])

        # Normalize sensor readings [0, 1], assuming max value is 4096
        norm_sensors = np.array(proximity_sensors) / 4096.0

        # Calculate weighted danger level
        danger_level = np.sum(norm_sensors * sensor_weights)

        # Detect front obstacles
        front_sensors = [proximity_sensors[0], proximity_sensors[1], proximity_sensors[7]]
        max_front = max(front_sensors)

        # Collision penalty
        if max_front > danger_threshold * 3:
            return 0.0  # Severe collision

        # Calculate safety score
        if max_front < danger_threshold:
            # Safe distance, high score
            safety_score = 1.0
        else:
            # Obstacle present, score based on distance
            safety_score = 1.0 - (max_front - danger_threshold) / (danger_threshold * 2)
            safety_score = max(0.2, safety_score)

        # Determine obstacle avoidance behavior
        left_obstacle = proximity_sensors[7] > danger_threshold
        right_obstacle = proximity_sensors[0] > danger_threshold
        front_obstacle = max_front > danger_threshold

        # Calculate line position: -1 (left), 0 (center), 1 (right), None (lost)
        line_position = None
        if center_ground < line_threshold:
            line_position = 0  # Line is centered
        elif left_ground < line_threshold:
            line_position = -1  # Line is on the left
        elif right_ground < line_threshold:
            line_position = 1  # Line is on the right

        avoidance_score = 0.5  # Default neutral score

        # CASE 1: Obstacle present - prioritize avoidance
        if front_obstacle or left_obstacle or right_obstacle:
            # Calculate speed difference for turning
            speed_diff = abs(left_speed - right_speed)

            if left_obstacle and not right_obstacle:
                # Left obstacle: should turn right (right wheel slower)
                if right_speed < left_speed:
                    avoidance_score = min(1.0, 0.5 + speed_diff / self.max_speed * 0.5)
                else:
                    avoidance_score = 0.2

            elif right_obstacle and not left_obstacle:
                # Right obstacle: should turn left (left wheel slower)
                if left_speed < right_speed:
                    avoidance_score = min(1.0, 0.5 + speed_diff / self.max_speed * 0.5)
                else:
                    avoidance_score = 0.2

            elif left_obstacle and right_obstacle:
                # Both sides blocked: should slow down or reverse
                avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
                if avg_speed < self.max_speed * 0.3:
                    avoidance_score = 0.8  # Good, slowing down
                else:
                    avoidance_score = 0.2  # Bad, going too fast

        # CASE 2: No immediate obstacle - try to return to line
        else:
            if line_position is not None:
                # Line detected, guide robot back to it
                if line_position == 0:
                    # Line is centered, go straight
                    if abs(left_speed - right_speed) < self.max_speed * 0.2:
                        avoidance_score = 1.0  # Perfect, going straight on line
                    else:
                        avoidance_score = 0.0

                elif line_position == -1:
                    # Line is on the left, should turn left gently
                    if left_speed < right_speed and (right_speed - left_speed) > 0.1:
                        # Turning left toward the line
                        avoidance_score = 0.9
                    else:
                        avoidance_score = 0.0  # Not turning toward line

                elif line_position == 1:
                    # Line is on the right, should turn right gently
                    if right_speed < left_speed and (left_speed - right_speed) > 0.1:
                        # Turning right toward the line
                        avoidance_score = 0.9
                    else:
                        avoidance_score = 0.0  # Not turning toward line
            else:
                # Line lost, encourage exploration (gentle turns)
                speed_diff = abs(left_speed - right_speed)
                if speed_diff > 0.1 and speed_diff < self.max_speed * 0.5:
                    avoidance_score = 0.6  # Gentle turning to search for line
                else:
                    avoidance_score = 0.0

        # Encourage speed reduction when obstacle is close
        if max_front > danger_threshold:
            avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
            speed_reduction = 1.0 - min(avg_speed / self.max_speed, 1.0)
            avoidance_score *= (0.7 + 0.3 * speed_reduction)

        # Combined fitness score
        # When obstacle is present: prioritize safety
        # When obstacle is clear: prioritize returning to line
        if front_obstacle:
            fitness = safety_score * 0.7 + avoidance_score * 0.3
        else:
            fitness = safety_score * 0.4 + avoidance_score * 0.6

        # Penalties
        if self.is_on_edge:
            fitness = 0.0

        if self.real_speed < 0.02 and max(abs(self.velocity_left), abs(self.velocity_right)) > 0.5:
            fitness = 0.0

        if abs(self.velocity_right) != 0:
            if abs(self.velocity_left) / abs(
                    self.velocity_right) > 0.8 and self.velocity_right * self.velocity_left < 0:
                fitness = 0.0

        if self.is_on_edge:
            fitness = 0.0

        return max(0.0, fitness)

    def spinningFitness(self):
        """
        Spinning penalty function

        Goal: Penalize in-place spinning and ineffective oscillation behavior

        Principle:
        - In-place spinning: Two wheels have equal speed in opposite directions
        - Continuous oscillation: Frequently changing turning direction
        - These behaviors waste time and don't help accomplish the task

        Args:
            left_speed: Left wheel speed
            right_speed: Right wheel speed
            angular_velocity_history: Historical angular velocity records

        Returns: Penalty score [0, 1], 1 means no penalty, 0 means maximum penalty

        Design points:
        1. Detect in-place spinning: Speeds are opposite and similar in magnitude
        2. Detect oscillation: Frequently changing turning direction
        3. Allow necessary turning: Small turns are not penalized
        4. Time penalty: Continuous spinning increases penalty
        """
        # Calculate angular velocity (simplified model)
        # Positive value indicates counterclockwise rotation, negative value indicates clockwise rotation
        self.action_number += 1
        # if(self.action_number%30==0):
        #     print("self.velocity_left:",self.velocity_left,"self.velocity_right:",self.velocity_right)
        left_speed, right_speed, angular_velocity_history = self.velocity_left, self.velocity_right, None
        angular_velocity = right_speed - left_speed

        # Detect in-place spinning
        speed_sum = abs(left_speed) + abs(right_speed)
        speed_diff = abs(abs(left_speed) - abs(right_speed))
        if self.real_speed < 0.0001:
            return 0.0

        if self.real_speed < 0.01 and max(abs(self.velocity_left), abs(self.velocity_right)) > 0.5:
            return 0.0
        if abs(self.velocity_right) != 0:
            if abs(self.velocity_left) / abs(
                    self.velocity_right) > 0.8 and self.velocity_right * self.velocity_left < 0:
                return 0.0

        if speed_sum < 0.1:
            # Almost stationary, no penalty
            return 1.0

        # In-place spinning detection: Speeds are opposite and similar in magnitude
        if left_speed * right_speed < 0:  # Opposite signs
            similarity = 1.0 - speed_diff / (speed_sum + 1e-6)
            if similarity > 0.8:
                # Obvious in-place spinning
                spinning_penalty = similarity
                return max(0.0, 1.0 - spinning_penalty * 0.8)

        # Detect oscillation behavior
        if angular_velocity_history and len(angular_velocity_history) > 5:
            recent_history = angular_velocity_history[-10:]

            # Count direction changes
            direction_changes = 0
            for i in range(1, len(recent_history)):
                if recent_history[i] * recent_history[i - 1] < 0:
                    direction_changes += 1

            # Frequent direction changes indicate oscillation
            if direction_changes > 5:
                oscillation_penalty = min(direction_changes / 10.0, 0.6)
                return max(0, 1.0 - oscillation_penalty)

        # if self.is_on_edge:
        #     return 0.0
        # print(max(abs(self.velocity_left), abs(self.velocity_right)))
        # Slight turning is not penalized
        turn_ratio = abs(angular_velocity) / (speed_sum + 1e-6)

        if turn_ratio < 0.3:
            return 1.0

        # Moderate turning gets slight penalty
        return max(0.0, 1.0 - turn_ratio * 0.2)

    def check_for_new_genes(self):
        if(self.flagMessage == True):
                # Split the list based on the number of layers of your network
                part = []
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        part.append((self.number_neuros_per_layer[n-1]+1)*(self.number_neuros_per_layer[n]))
                    else:   
                        part.append(self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n])
                
                # Set the weights of the network
                data = []
                weightsPart = []
                sum = 0
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        weightsPart.append(self.receivedData[n-1:part[n-1]])
                    elif(n == (len(self.number_neuros_per_layer)-1)):
                        weightsPart.append(self.receivedData[sum:])
                    else:
                        weightsPart.append(self.receivedData[sum:sum+part[n-1]])
                    sum += part[n-1]
                for n in range(1,len(self.number_neuros_per_layer)):  
                    if(n == 1):
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1]+1,self.number_neuros_per_layer[n]])    
                    else:
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1],self.number_neuros_per_layer[n]])    
                    data.append(weightsPart[n-1])                
                self.network.weights = data
                
                #Reset fitness list
                self.fitness_values = []
        
    def clip_value(self,value,min_max):
        if (value > min_max):
            return min_max
        elif (value < -min_max):
            return -min_max
        return value

    def sense_compute_and_actuate(self):
        # MLP: 
        #   Input == sensory data
        #   Output == motors commands
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]

        
        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left*3)
        self.right_motor.setVelocity(self.velocity_right*3)

    def calculate_fitness(self):
        # pos = self.robot.getSelf().getPosition()
        # print("Robot Position: x {:.3f} y {:.3f} z {:.3f}".format(pos[0],pos[1],pos[2]))
        #
        ### Define the fitness function to increase the speed of the robot and 
        ### to encourage the robot to move forward only
        forwardFitness = self.forwardFitness()
        
        ### Define the fitness function to encourage the robot to follow the line
        followLineFitness = self.followLineFitness()
                
        ### Define the fitness function to avoid collision
        avoidCollisionFitness = self.avoidCollisionFitness()
        
        ### Define the fitness function to avoid spining behaviour
        spinningFitness = self.spinningFitness()



        if self.current_generation<=0.3*self.num_generations:
            fitnessWeightsMapping = {"forwardFitness":0.55,"followLineFitness":0.2,"avoidCollisionFitness":0.2,"spinningFitness":0.05}
        elif self.current_generation > 0.3 * self.num_generations and self.current_generation <= 0.7 * self.num_generations:
            fitnessWeightsMapping = {"forwardFitness": 0.25, "followLineFitness": 0.4, "avoidCollisionFitness": 0.2,
                                     "spinningFitness": 0.05}
        elif self.current_generation > 0.7 * self.num_generations and self.current_generation <=  self.num_generations:
            fitnessWeightsMapping = {"forwardFitness": 0.20, "followLineFitness": 0.4, "avoidCollisionFitness": 0.35,
                                     "spinningFitness": 0.05}
        # if self.action_number % 100 == 0:
        #     print("num_generatetions:", self.num_generations, " current_generation:", self.current_generation)
        #     print("Fitness Weights Mapping:", fitnessWeightsMapping)


        ### Define the fitness function of this iteration which should be a combination of the previous functions         
        combinedFitness = forwardFitness*fitnessWeightsMapping['forwardFitness'] + followLineFitness*fitnessWeightsMapping['followLineFitness'] + avoidCollisionFitness*fitnessWeightsMapping['avoidCollisionFitness'] + spinningFitness*fitnessWeightsMapping['spinningFitness']
        # if self.action_number % 100 == 0:
        #     print("Fitness Components - Forward: {:.3f}, Line: {:.3f}, Avoid Collision: {:.3f}, Spinning Penalty: {:.3f}".format(forwardFitness, followLineFitness, avoidCollisionFitness, spinningFitness))
        #     print("real speed:",self.real_speed)
        #     print("combinedFiteness:,",combinedFitness)
        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values) 

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send fitness:", string_message)
        self.emitter.send(string_message)
            
    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while(self.receiver.getQueueLength() > 0):
                # Adjust the Data to our model                
                # Webots 2022: 
                # self.receivedData = self.receiver.getData().decode("utf-8")
                # Webots 2023: 
                data_from_supervisor = self.receiver.getString()
                # print("robot received:", self.receivedData)
                # print(type(self.receivedData))
                if data_from_supervisor.startswith("genotype: "):
                    # print(data_from_supervisor)
                    self.receivedData = data_from_supervisor[11:-1]
                    self.receivedData = self.receivedData.split()
                    x = np.array(self.receivedData)
                    self.receivedData = x.astype(float)
                    # print("Controller handle receiver data:", self.receivedData)
                elif data_from_supervisor.startswith("current_generation: "):
                    generation_data = data_from_supervisor[20:len(data_from_supervisor)]
                    # print("Received generation data:", data_from_supervisor)
                    self.current_generation = int(generation_data)
                    # print("Controller handle receiver generation:", self.current_generation)
                elif data_from_supervisor.startswith("num_generations: "):
                    num_generations = data_from_supervisor[17:len(data_from_supervisor)]
                    self.num_generations = int(num_generations)
                    # print("Controller handle receiver population:", self.num_generations)
                elif data_from_supervisor.startswith("real_speed: "):
                    speed_data = data_from_supervisor[12:len(data_from_supervisor)]
                    self.real_speed = float(speed_data)
                    # print("Controller handle receiver real speed:", self.real_speed)
                elif data_from_supervisor.startswith("position: "):
                    position_data = data_from_supervisor[10:len(data_from_supervisor)]
                    # print("Received position data:", position_data)
                    # Convert string representation of list to actual list
                    pos = eval(position_data)
                    x, y, z = pos
                    if abs(x) > 0.69 or abs(y) > 0.69:
                        self.is_on_edge = True
                        # if self.is_on_edge:
                        #     if self.action_number % 100 == 0:
                        #         print("x,y:{},{}".format(x,y))
                    # print("Controller handle receiver position:", position_list)
                    else:
                        self.is_on_edge = False
                self.receiver.nextPacket()
                
            # Is it a new Genotype?
            if(np.array_equal(self.receivedDataPrevious,self.receivedData) == False):
                self.flagMessage = True
                
            else:
                self.flagMessage = False
                
            self.receivedDataPrevious = self.receivedData 
        else:
            #print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):        
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []
            
            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()
            
            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            #print("Ground Sensors \n    left {} center {} right {}".format(left,center,right))
                        
            ### Please adjust the ground sensors values to facilitate learning 
            min_gs = 0
            max_gs = 100
            
            if(left > max_gs): left = max_gs
            if(center > max_gs): center = max_gs
            if(right > max_gs): right = max_gs
            if(left < min_gs): left = min_gs
            if(center < min_gs): center = min_gs
            if(right < min_gs): right = min_gs
            
            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left-min_gs)/(max_gs-min_gs))
            self.inputs.append((center-min_gs)/(max_gs-min_gs))
            self.inputs.append((right-min_gs)/(max_gs-min_gs))
            #print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))
            
            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if(i==0 or i==1 or i==2 or i==3 or i==4 or i==5 or i==6 or i==7):        
                    temp = self.proximity_sensors[i].getValue()
                    
                    ### Please adjust the distance sensors values to facilitate learning 
                    min_ds = 0
                    max_ds = 100
                    
                    if(temp > max_ds): temp = max_ds
                    if(temp < min_ds): temp = min_ds
                    
                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append((temp-min_ds)/(max_ds-min_ds))
                    #print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))
    
            # GA Iteration       
            # Verify if there is a new genotype to be used that was sent from Supervisor  
            self.check_for_new_genes()
            # Define the robot's actuation (motor values) based on the output of the MLP 
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()
            
            # End of the iteration

            
if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
