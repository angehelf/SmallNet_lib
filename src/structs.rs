use crate::activation_function::*;

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