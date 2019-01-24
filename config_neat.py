[NEAT]
fitness_criterion     = mean 
fitness_threshold     = 10000
pop_size              = 10
reset_on_extinction   = True

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.1
activation_options      = sigmoid gauss relu tanh
#abs clamped cube exp gauss hat identity inv log relu sigmoid sin softplus square tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.05
aggregation_options     = sum product min max mean median maxabs

# node bias options
bias_init_mean          = 0.05
bias_init_stdev         = 1.0
bias_max_value          = 20.0
bias_min_value          = -20.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.2
bias_replace_rate       = 0.3

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.3

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.5

feed_forward            = True
#initial_connection      = unconnected
initial_connection      = partial_nodirect 0.5

# node add/remove rates
node_add_prob           = 0.5
node_delete_prob        = 0.3

# network parameters
num_hidden              = 1
num_inputs              = 13
num_outputs             = 8

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.05
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.3
response_mutate_rate    = 0.75
response_replace_rate   = 0.1

# connection weight options
weight_init_mean        = 0.1
weight_init_stdev       = 1.0
weight_max_value        = 20
weight_min_value        = -20
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.3

[DefaultSpeciesSet]
compatibility_threshold = 2.5

[DefaultStagnation]
species_fitness_func = max 
species_elitism      = 0

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.3
