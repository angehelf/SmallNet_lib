use std::{iter::FlatMap, vec};

use crate::activation_function::*;
use log::{debug, info, warn};
pub mod activation_function;
use rand::{RngExt, random_range};
pub struct SmallNet {
    pub layers_list: Vec<Layer>,
}
#[derive(Clone)]
pub struct Layer {
    pub neurons_list: Vec<Neuron>,
}
pub enum ActivationInitType {
    Unique,
    PerLayer,
    Random,
}
pub enum ConnectionInitType {
    FullyConnected,
    RandomConnection,
    RandomConnectionWithLayerSkiping,
}
#[derive(PartialEq, Eq, Copy, Clone)]
pub struct NeuronId {
    pub layer: usize,
    pub position: usize,
}
#[derive(Clone, Copy)]
pub struct Connection {
    pub source_neuron_id: NeuronId,
    pub souce_value: f32,
    pub connection_weight: f32,
}
#[derive(Clone)]
pub struct Neuron {
    pub neuron_id: NeuronId,
    pub activation_function: fn(f32) -> f32,
    pub connections_list: Vec<Connection>,
    pub activation_value: f32,
    pub bias: f32,
}
impl Default for Neuron {
    fn default() -> Self {
        Self {
            neuron_id: NeuronId {
                layer: usize::MAX,
                position: usize::MAX,
            },
            activation_function: linear,
            connections_list: Vec::new(),
            activation_value: 0.0,
            bias: 0.0,
        }
    }
}
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

    pub fn get_mut_neuron_from_id(&mut self, id: NeuronId) -> &mut Neuron {
        &mut self.layers_list[id.layer as usize].neurons_list[id.position as usize]
    }

    pub fn get_neuron_from_id(&self, id: NeuronId) -> &Neuron {
        &self.layers_list[id.layer as usize].neurons_list[id.position as usize]
    }

    pub fn get_output(&self) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::new();
        for neuron in &self.layers_list.last().unwrap().neurons_list {
            result.push(neuron.activation_value.clone());
        }

        return result;
    }

    pub fn neurons_iterator_mut(&mut self) -> impl Iterator<Item = &mut Neuron> {
        self.layers_list
            .iter_mut()
            .flat_map(|layer| layer.neurons_list.iter_mut())
    }
    pub fn copy_neurons_in_layer(layer: &Layer)->Vec<Neuron>{
        layer.neurons_list.clone()
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

    pub fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        if inputs.len() != self.layers_list[0].neurons_list.len() {
            println!("le vecteurs d'inputs n'a pas la bonne taille");
            return Vec::new();
        }
        for neuron in self.layers_list[0].neurons_list.iter_mut() {
            neuron.activation_value = inputs[neuron.neuron_id.position];
        }

        

        for _i in 1..self.layers_list.len(){
           
            let copied_neuron_list = SmallNet::copy_neurons_in_layer(&self.layers_list[_i-1]);
            for neuron in &mut self.layers_list[_i].neurons_list{

                 neuron.activation_value=0.0;

                for connection in &mut neuron.connections_list{

                    neuron.activation_value += connection.connection_weight * copied_neuron_list[connection.source_neuron_id.position].activation_value;
                    
                }

                neuron.activation_value+= neuron.bias;
                neuron.activation_value= (neuron.activation_function)(neuron.activation_value);
            }

        }


        self.get_output()
    }
 
}

