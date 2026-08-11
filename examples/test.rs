use small_net_lib::*;
use small_net_lib::activation_function::*;
use rand::*;
use rand_distr::{Normal,Distribution};
use std::time::Instant;
fn main(){

let normal = Normal::new(0.0,1.0);
let mut rng = rand::rng();
let mut test = SmallNet::new_grid(vec![3,8,3]);
test.initialize_activation_functions(ActivationInitType::PerLayer, vec![relu,tanh]);


test.initialize_connections(ConnectionInitType::FullyConnected,|| normal.unwrap().sample(&mut rng));
test.initialize_bias(|| normal.unwrap().sample(&mut rng));
//let a = test.get_neuron_from_id(NeuronId { layer: 1, position: 2 }).connections_list[0].connection_weight;
let inputs = vec![0.25,0.75,-0.25];

let start = Instant::now();
let mut a = Vec::default();
for _i in 0..1000{
a = test.feed_forward(&inputs);
}
let duration = start.elapsed();

println!("{:?}",a);
println!("durée d'execution: {:?}",duration);


}

//