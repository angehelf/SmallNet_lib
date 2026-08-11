use crate::structs::*;
use crate::activation_function::*;
use rand::*;
impl SmallNet {
    ///Génère une nouvelles grille vierge de neurone (sans aucune connection ni fonction d'activation)
    /// # Arguement
    /// * 'grid' - chaque valeur stoquée dans le vec représente le nombre de neurones par couches du résaux.
    /// # Returns
    /// un noueau résaux vièrge de connection.
    pub fn new_grid(grid: Vec<usize>) -> Self {
        let mut layers_list = Vec::new();
        for i in 0..grid.len() {
            layers_list.push(Layer {
                neurons_list: Vec::new(),
            });

            for j in 0..grid[i] {
                layers_list[i].neurons_list.push(Neuron {
                    neuron_id: NeuronId {
                        layer: i,
                        position: j,
                    },
                    activation_function: linear,
                    connections_list: Vec::new(),
                    activation_value: 0.0,
                    bias: 0.0,
                })
            }
        }
        println!("le résaux à été généré avec succes");
        Self { layers_list }
    }

    ///Assigne la fonction d'activation des neurones du résaux dans son entièrté en fonction des arguments fournis
    /// #Arguments
    /// *'activation_type' est un énum permetant de determiné le comportement d'atribution
    /// *'function_list' est la liste de fonction a fournir
    pub fn initialize_activation_functions(
        &mut self,
        activation_type: ActivationInitType,
        function_list: Vec<fn(f32) -> f32>,
    ) {
        match activation_type {
            ActivationInitType::Unique => {
                if function_list.len() != 1 {
                    println!(
                        "(initialize_activation_functions)plusieurs fonction passé en arguments pour une initialisation de type unique"
                    );
                    return;
                } else {
                    for neuron in self.neurons_iterator_mut() {
                        if neuron.neuron_id.layer != 0 {
                            neuron.activation_function = function_list[0]
                        }
                    }
                }
            }

            ActivationInitType::PerLayer => {
                if function_list.len() != self.layers_list.len()-1 {
                    println!(
                        "(initialize_activation_functions) la longueur de la liste de fonction ne corepond pas au nombres de couche du résaux"
                    );
                    return;
                } else {
                    for neuron in self.neurons_iterator_mut().filter(|n| n.neuron_id.layer !=0) {
                        let index = neuron.neuron_id.layer-1;
                        neuron.activation_function = function_list[index];
                    }
                }
            }
            ActivationInitType::Random => {
                if function_list.len() < 2 {
                    println!(
                        "(initialize_activation_functions) moins de deux fonction passé en argument pour une séléction aleatoire"
                    );
                } else {
                    let mut rng = rand::rng();
                    let range = function_list.len();
                    for neuron in &mut self.neurons_iterator_mut() {
                        neuron.activation_function = function_list[rng.random_range(0..range)]
                    }
                }
            }
        }
    }

    pub fn initialize_connections<F>(
        &mut self,
        connection_init_type: ConnectionInitType,
        mut init: F,
    ) where
        F: FnMut() -> f32,
    {
        match connection_init_type {
            ConnectionInitType::FullyConnected => {
                //parcours tous les neurones hormis ceux situé dans la 1er couche (couche 0) du résaux
                for layer in 1..self.layers_list.len() {
                    let (part1, part2) = self.layers_list.split_at_mut(layer);

                    for neuron in &mut part2[0].neurons_list {
                        for prev_neuron in &part1[layer - 1].neurons_list {
                            neuron.connections_list.push(Connection {
                                source_neuron_id: prev_neuron.neuron_id,
                                souce_value: prev_neuron.activation_value,
                                connection_weight: init(),
                            });
                        }
                    }
                }
            }
            ConnectionInitType::RandomConnection => {}
            ConnectionInitType::RandomConnectionWithLayerSkiping => {}
        }
    }

    pub fn initialize_bias<F>(&mut self, mut init: F)
    where
        F: FnMut() -> f32,
    {
        for neuron in self
            .neurons_iterator_mut()
            .filter(|x| x.neuron_id.layer != 0)
        {
            neuron.bias = init();
        }
    }
}