use crate::structs::SmallNet;

impl SmallNet {


   
    

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
