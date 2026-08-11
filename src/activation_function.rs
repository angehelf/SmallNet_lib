const ELU_ALPHA : f32 = 0.1;
pub fn linear(input : f32)->f32{


    input
  
}

pub fn relu(input:f32)->f32{

if input >0.0 {return input;}

else{
    return 0.0;
}
}

pub fn elu(input:f32)->f32{

if input >0.0 {return input;}

else{
    return ELU_ALPHA*(f32::exp(input)-1.0)
}
}

pub fn tanh(input:f32)->f32{

f32::tanh(input)
}

pub fn sigmoid(input:f32)->f32{

1.0/(1.0+f32::exp(-input))
}
