extern crate blas_src;

mod web;

use std::{f64, fs, path::Path};

use clap::Parser;
use rand::rng;
use rand::seq::SliceRandom;
use mnist::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::prelude::IteratorRandom;
use serde::{Deserialize, Serialize};
use web::PredictionResponse;


#[derive(Serialize, Deserialize)]
struct Network {
    performance: f64,
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
            let weight_matrix = Array2::<f64>::random(
                (shape[i - 1], shape[i]),
                Uniform::new(
                    -1.0 / (shape[i - 1] as f64).sqrt(),
                    1.0 / (shape[i - 1] as f64).sqrt(),
                ),
            );
            let bias_matrix = Array2::<f64>::random((1, shape[i]), uniform.clone());
            weights.push(weight_matrix);
            biases.push(bias_matrix);
        }

        Network {
            shape,
            biases,
            weights,
            performance: 0.0,
        }
    }

    fn feedforward(&self, mut a: Array2<f64>) -> Array2<f64> {
        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            let weighted_input = a.dot(weight) + bias;
            a = sigmoid(&weighted_input);
        }
        a
    }

    fn stochastic_gradient_descent(
        &mut self,
        mut training_data: Vec<(Array1<f64>, Array1<f64>)>,
        epochs: usize,
        batch_size: usize,
        training_rate: f64,
        test_data: Option<Vec<(Array1<f64>, Array1<f64>)>>,
    ) {
        for i in 0..epochs {
            training_data.shuffle(&mut rng());
            for (batch_index, batch) in training_data.chunks(batch_size).enumerate() {
                self.update_batch(batch, training_rate);
            }

            if test_data.is_some() {
                let res = self.evaluate(test_data.clone().unwrap());
                self.performance = res;
                println!(
                    "Epoch {}: {}/{}\t{:.2}%",
                    i,
                    res * 10_000.0,
                    10_000.0,
                    res * 100.0
                );
            } else {
                println!("Epoch {} complete", i);
            }
            save_network(self);
        }
    }

    fn update_batch(&mut self, batch: &[(Array1<f64>, Array1<f64>)], training_rate: f64) {
        let mut nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::<f64>::zeros(b.raw_dim()))
            .collect();

        let mut nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::<f64>::zeros(w.raw_dim()))
            .collect();

        let (delta_nabla_b, delta_nabla_w) = self.backprop(batch);
        nabla_b
            .iter_mut()
            .zip(delta_nabla_b.iter())
            .for_each(|(a, b)| *a += b);
        nabla_w
            .iter_mut()
            .zip(delta_nabla_w.iter())
            .for_each(|(a, b)| *a += b);

        let scaling_factor = training_rate / batch.len() as f64;
        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
            *w -= &nw.mapv(|x| x * scaling_factor);
        }
        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter()) {
            *b -= &nb.mapv(|x| x * scaling_factor);
        }
    }

    fn backprop(
        &self,
        batch: &[(Array1<f64>, Array1<f64>)],
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|b| Array2::<f64>::zeros(b.raw_dim()))
            .collect();

        let mut nabla_w: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::<f64>::zeros(w.raw_dim()))
            .collect();

        let mut activation = Array2::<f64>::zeros((batch.len(), self.shape[0]));
        let mut classifications =
            Array2::<f64>::zeros((batch.len(), self.shape[self.shape.len() - 1]));
        for (i, (img, label)) in batch.iter().enumerate() {
            activation.slice_mut(s![i, ..]).assign(img);
            classifications.slice_mut(s![i, ..]).assign(label);
        }

        let mut weighted_inputs = Vec::new();
        let mut activations = vec![activation.clone()];
        // Forward pass
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let weighted_input = activation.dot(w) + b;
            activation = sigmoid(&weighted_input);
            weighted_inputs.push(weighted_input.clone());
            activations.push(activation.clone());
        }

        let mut delta = self.cost_derivative(activations.last().unwrap(), &classifications)
            * sigmoid_prime(weighted_inputs.last().unwrap());

        let last_layer = self.weights.len() - 1;
        nabla_b[last_layer]
            .slice_mut(s![0, ..])
            .assign(&delta.sum_axis(Axis(0)));
        nabla_w[last_layer] = activations[activations.len() - 2].t().dot(&delta);

        // Backpropagate through hidden layers
        for l in (0..last_layer).rev() {
            let z = &weighted_inputs[l];
            let sp = sigmoid_prime(z);
            delta = delta.dot(&self.weights[l + 1].t()) * sp;

            nabla_b[l]
                .slice_mut(s![0, ..])
                .assign(&delta.sum_axis(Axis(0)));
            nabla_w[l] = activations[l].t().dot(&delta);
        }

        (nabla_b, nabla_w)
    }

    fn evaluate(&self, test_data: Vec<(Array1<f64>, Array1<f64>)>) -> f64 {
        let test_results: Vec<(usize, usize)> = test_data
            .iter()
            .map(|(x, y)| {
                let result = self.feedforward(x.clone().insert_axis(Axis(0)));
                (
                    result
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0),
                    y.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0),
                )
            })
            .collect();

        test_results.iter().filter(|(x, y)| x == y).count() as f64 / test_results.len() as f64
    }

    fn cost_derivative(&self, output_activations: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        output_activations - y
    }

    pub fn get_random_predictions(&self, test_data: &Vec<(Array1<f64>, Array1<f64>)>) -> Vec<PredictionResponse> {
        /* let random_samples = test_data.choose_multiple(&mut rng, 10).cloned().collect(); */
        let random_samples = test_data.iter().choose_multiple(&mut rng(), 10);
        random_samples
            .into_iter()
            .map(|(image, true_label)| {
                let output = self.feedforward(image.clone().insert_axis(Axis(0)));
                let predicted_label = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u8)
                    .unwrap_or(0);

                PredictionResponse {
                    image: image.to_vec(),
                    predicted_label,
                    true_label: true_label.iter().position(|&x| x == 1.0).unwrap_or(0) as u8,
                    confidences: output.iter().map(|&x| x as f32).collect(),
                    is_correct: predicted_label == true_label.iter().position(|&x| x == 1.0).unwrap_or(0) as u8,
                }
            })
            .collect()
    }

    pub fn get_misclassified(&self, test_data: &Vec<(Array1<f64>, Array1<f64>)> ) -> Vec<PredictionResponse> {
        test_data
            .iter()
            .map(|(image, true_label)| {
                let output = self.feedforward(image.clone().insert_axis(Axis(0)));
                let predicted_label = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u8)
                    .unwrap_or(0);
                let true_index = true_label.iter().position(|&x| x == 1.0).unwrap_or(0) as u8;
                
                PredictionResponse {
                    image: image.to_vec(),
                    predicted_label,
                    true_label: true_index,
                    confidences: output.iter().map(|&x| x as f32).collect(),
                    is_correct: predicted_label == true_index,
                }
            })
            .filter(|pred| !pred.is_correct).choose_multiple(&mut rng(), 10)

    }
}

fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + (-z).mapv(f64::exp))
}

fn sigmoid_prime(z: &Array2<f64>) -> Array2<f64> {
    sigmoid(z) * (1.0 - sigmoid(z))
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Shape of the network as a comma-separated list (e.g., 784,30,10)
    #[arg(short, long)]
    shape: Option<String>,

    /// Name of a network to load excluding .json
    #[arg(short = 'l', long)]
    load: Option<String>,

    /// Flag for listing available networks
    #[arg(short = 'L', long, default_value_t = false)]
    list: bool,

    /// Epochs to train the network
    #[arg(short, long, default_value = "30")]
    epochs: usize,

    /// Batch size for training the network
    #[arg(short, long, default_value = "10")]
    batch_size: usize,

    /// Training rate for the network
    #[arg(short, long, default_value = "0.01")]
    training_rate: f64,

    /// Start the web viewer
    #[arg(short, long, default_value_t = false)]
    web: bool,
}

fn main() {
    let args: Args = Args::parse();

    if args.list {
        list_networks();
        return;
    }

    let mut network: Network;

    if let Some(load_name) = args.load {
        if Path::new(&load_name).exists() {
            network = load_network(&load_name);
            println!("Loaded existing network from {}", load_name);
        } else {
            eprintln!("Network file not found: {}", load_name);
            return;
        }
    } else if let Some(shape_str) = args.shape {
        let shape: Vec<usize> = shape_str
            .split(',')
            .map(|s| s.parse().expect("Invalid network shape"))
            .collect();

        if (shape.len() < 2) {
            eprintln!("Network shape must have at least two layers.");
            return;
        }

        let model_dir = Path::new("./models");
        if model_dir.exists() {
            let entries = fs::read_dir(model_dir).expect("Unable to read model directory");
            for entry in entries.filter_map(Result::ok) {
                if entry
                    .file_name()
                    .into_string()
                    .unwrap()
                    .starts_with(&shape_str)
                    && entry.file_name().into_string().unwrap().ends_with(".json")
                {
                    eprintln!("A network with the specified shape already exists. Archive it or delete it.");
                    return;
                }
            }
        }

        println!("Creating a new network with shape {:?}", shape);
        network = Network::new(shape);
        save_network(&network);
    } else {
        eprintln!("Either shape or load argument is required.");
        return;
    }

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

    let mut train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

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

    if args.web {
        println!("Starting web interface...");
        web::start_server(network, test_data_tuples).expect("Failed to start web server");
    } else {
        network.stochastic_gradient_descent(
            train_data_tuples,
            args.epochs,
            args.batch_size,
            args.training_rate,
            Some(test_data_tuples),
        );
    }
}

fn load_network(path: &str) -> Network {
    let json = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&json).unwrap()
}

fn save_network(network: &Network) {
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let shape_str = network
        .shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<String>>()
        .join(",");
    let file_name = format!(
        "./models/{}-{}-{:.2}.json",
        shape_str, date, network.performance
    );

    let model_dir = Path::new("./models");
    if let Ok(entries) = fs::read_dir(model_dir) {
        for entry in entries.filter_map(Result::ok) {
            if entry
                .file_name()
                .into_string()
                .unwrap()
                .starts_with(&format!("{}-{}", shape_str, date))
                && entry.file_name().into_string().unwrap().ends_with(".json")
            {
                fs::remove_file(entry.path()).expect("Failed to delete old network file");
            }
        }
    }

    let json = serde_json::to_string(network).unwrap();
    fs::write(&file_name, json).expect("Unable to save network");
    println!("Network saved to {}", file_name);
}

fn list_networks() {
    let model_dir = Path::new("./models");

    if !model_dir.exists() {
        println!("Model directory does not exist.");
        return;
    }

    let entries = match fs::read_dir(model_dir) {
        Ok(entries) => entries.filter_map(Result::ok).collect::<Vec<_>>(),
        Err(_) => {
            println!("Unable to read model directory");
            return;
        }
    };

    let json_files: Vec<_> = entries
        .iter()
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
        .collect();

    if json_files.is_empty() {
        println!("No model files found.");
    } else {
        println!("Available models:");
        for entry in json_files {
            println!("{}", entry.file_name().to_string_lossy());
        }
    }
}
