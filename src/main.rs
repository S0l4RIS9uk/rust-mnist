extern crate blas_src;

use std::f64;

use rand::rng;
use rand::seq::SliceRandom;

use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt; // Make sure RandomExt is imported
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
struct Network {
    shape: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(shape: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let uniform = Uniform::new(0.0, 1.0);

        for i in 1..shape.len() {
            let weight_matrix = Array2::<f64>::random((shape[i], shape[i - 1]), uniform.clone());
            let bias_matrix = Array2::<f64>::random((shape[i], 1), uniform.clone());
            weights.push(weight_matrix);
            biases.push(bias_matrix);
        }

        Network {
            shape,
            biases,
            weights,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut network = Network::new(args[1].split(',').map(|x| x.parse().unwrap()).collect());
    save_network(&network);
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let mut train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f64> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);

    let test_labels: Array2<f64> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    fn one_hot_encode(label: f64) -> Array1<f64> {
        let mut encoded = Array1::<f64>::zeros(10);
        encoded[label as usize] = 1.0;
        encoded
    }

    let train_data_tuples: Vec<(Array1<f64>, Array1<f64>)> = train_data
        .outer_iter()
        .zip(train_labels.iter())
        .map(|(image, &label)| {
            let flattened_image = image.into_shape(784).unwrap().to_owned();
            let one_hot_label = one_hot_encode(label);
            (flattened_image, one_hot_label)
        })
        .collect();

    let test_data_tuples: Vec<(Array1<f64>, Array1<f64>)> = test_data
        .outer_iter()
        .zip(test_labels.iter())
        .map(|(image, &label)| {
            let flattened_image = image.into_shape(784).unwrap().to_owned();
            let one_hot_label = one_hot_encode(label);
            (flattened_image, one_hot_label)
        })
        .collect();

    stochastic_gradient_descent(
        &mut network,
        train_data_tuples,
        30,
        10,
        3.0,
        Some(test_data_tuples),
    );
}

fn feedforward(network: &Network, mut a: Array2<f64>) -> Array2<f64> {
    for (weight, bias) in network.weights.iter().zip(network.biases.iter()) {
        let weighted_input = weight.dot(&a) + bias;
        a = sigmoid(&weighted_input);
    }
    a
}

fn stochastic_gradient_descent(
    network: &mut Network,
    mut training_data: Vec<(Array1<f64>, Array1<f64>)>,
    epochs: usize,
    batch_size: usize,
    training_rate: f64,
    test_data: Option<Vec<(Array1<f64>, Array1<f64>)>>,
) {
    for i in 0..epochs {
        training_data.shuffle(&mut rng());
        for batch in training_data.chunks(batch_size) {
            update_batch(network, batch, training_rate);
        }

        if test_data.is_some() {
            println!(
                "Epoch {}: {}/{}\t{:.2}%",
                i,
                evaluate(network, test_data.clone().unwrap()) * 10_000.0,
                10_000.0,
                evaluate(network, test_data.clone().unwrap()) * 100.0
            );
        } else {
            println!("Epoch {} complete", i);
        }
        save_network(network);
    }
}

fn update_batch(network: &mut Network, batch: &[(Array1<f64>, Array1<f64>)], training_rate: f64) {
    let mut nabla_b: Vec<Array2<f64>> = network
        .biases
        .iter()
        .map(|b| Array2::<f64>::zeros(b.raw_dim())) // Ensure shape (y, 1)
        .collect();

    let mut nabla_w: Vec<Array2<f64>> = network
        .weights
        .iter()
        .map(|w| Array2::<f64>::zeros(w.raw_dim())) // Ensure shape (y, x)
        .collect();

    let scaling_factor = training_rate / batch.len() as f64;
    for (x, y) in batch {
        let (delta_nabla_b, delta_nabla_w) = backprop(network, x.clone(), y.clone());

        nabla_b = nabla_b
            .iter()
            .zip(delta_nabla_b.iter())
            .map(|(a, b)| a + b)
            .collect();
        nabla_w = nabla_w
            .iter()
            .zip(delta_nabla_w.iter())
            .map(|(a, b)| a + b)
            .collect();

        network.weights = network
            .weights
            .iter()
            .zip(nabla_w.iter())
            .map(|(a, b)| a - b * scaling_factor)
            .collect();
        network.biases = network
            .biases
            .iter()
            .zip(nabla_b.iter())
            .map(|(a, b)| a - b * scaling_factor)
            .collect();
    }
}

fn backprop(
    network: &Network,
    x: Array1<f64>,
    y: Array1<f64>,
) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let mut nabla_b: Vec<Array2<f64>> = network
        .biases
        .iter()
        .map(|b| Array2::<f64>::zeros(b.raw_dim())) // Ensure shape (y, 1)
        .collect();

    let mut nabla_w: Vec<Array2<f64>> = network
        .weights
        .iter()
        .map(|w| Array2::<f64>::zeros(w.raw_dim())) // Ensure shape (y, x)
        .collect();

    let nabla_b_len = nabla_b.len();
    let nabla_w_len = nabla_w.len();
    let mut activation = x.insert_axis(Axis(1));
    let mut activations = vec![activation.clone()];

    let mut weighted_inputs = Vec::new();

    for (b, w) in network.biases.iter().zip(network.weights.iter()) {
        let weighted_input = w.dot(&activation) + b;
        weighted_inputs.push(weighted_input.clone());

        activation = sigmoid(&weighted_input);
        activations.push(activation.clone());
    }

    let activations_len = activations.len();
    let mut delta = cost_derivative(&activations.last().unwrap(), &y)
        * sigmoid_prime(&weighted_inputs.last().unwrap());

    nabla_b[nabla_b_len - 1] = delta.clone();
    let prev_activation = &activations[activations_len - 2];
    nabla_w[nabla_w_len - 1] = delta.dot(&prev_activation.t());

    for l in (1..network.weights.len()).rev() {
        let weighted_input = &weighted_inputs[l - 1];
        let sp = sigmoid_prime(weighted_input);
        delta = network.weights[l].t().dot(&delta) * sp;

        nabla_b[l - 1] = delta.clone();
        let prev_activation = &activations[l - 1];
        nabla_w[l - 1] = delta.dot(&prev_activation.t());
    }

    (nabla_b, nabla_w)
}

fn evaluate(network: &Network, test_data: Vec<(Array1<f64>, Array1<f64>)>) -> f64 {
    let test_results: Vec<(usize, usize)> = test_data
        .iter()
        .map(|(x, y)| {
            let result = feedforward(network, x.clone().into_shape((784, 1)).unwrap());
            (
                result
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0),
                y.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0),
            )
        })
        .collect();

    test_results.iter().filter(|(x, y)| x == y).count() as f64 / test_results.len() as f64
}

fn sigmoid(z: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn sigmoid_prime(z: &Array2<f64>) -> Array2<f64> {
    let sig = sigmoid(z);
    &sig * (1.0 - &sig)
}

fn cost_derivative(output_activations: &Array2<f64>, y: &Array1<f64>) -> Array2<f64> {
    output_activations - y.clone().insert_axis(Axis(1))
}

fn load_network(path: String) -> Network {
    let json = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&json).unwrap()
}

fn save_network(network: &Network) {
    let shape = network
        .shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<String>>()
        .join(",");
    let json = serde_json::to_string(&network).unwrap();
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    std::fs::write(format!("./models/{}-{}.json", shape, date), json).unwrap();
}
