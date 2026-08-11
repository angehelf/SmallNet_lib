 use crate::structs::*;
 
 impl SmallNet{
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

}