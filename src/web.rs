use axum::{
    routing::get,
    Router,
    Json,
    extract::State,
    response::IntoResponse,
    http::{StatusCode, Response, HeaderValue},
};
use serde::Serialize;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tokio;
use crate::Network;
use ndarray::Array1; // Assuming you are using ndarray

#[derive(Serialize)]
pub struct PredictionResponse {
    pub image: Vec<f64>,
    pub predicted_label: u8,
    pub true_label: u8,
    pub confidences: Vec<f32>,
    pub is_correct: bool,
}

#[derive(Clone)]
// Refactor the State struct to hold Arc<Network> and training_data
struct AppState {
    pub network: Arc<Network>,
    pub training_data: Vec<(Array1<f64>, Array1<f64>)>,
}

pub fn start_server(
    network: Network, 
    training_data: Vec<(Array1<f64>, Array1<f64>)>
) -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Runtime::new()?.block_on(async {
        let app_state = AppState {
            network: Arc::new(network),
            training_data,
        };

        let cors = CorsLayer::new()
            .allow_origin("*".parse::<HeaderValue>().unwrap());

        let app = Router::new()
            .route("/api/random", get(random_predictions))
            .route("/api/misclassified", get(misclassified_predictions))
            .route("/", get(serve_index))
            .layer(cors)
            .with_state(app_state);

        println!("Web interface running at http://localhost:3000");
        axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
            .serve(app.into_make_service())
            .await?;
        
        Ok(())
    })
}

async fn random_predictions(
    State(state): State<AppState>
) -> Json<Vec<PredictionResponse>> {
    Json(state.network.get_random_predictions(&state.training_data))
}

async fn misclassified_predictions(
    State(state): State<AppState>
) -> Json<Vec<PredictionResponse>> {
    Json(state.network.get_misclassified(&state.training_data))
}

async fn serve_index() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/html")
        .body(include_str!("../web/index.html").to_string())
        .unwrap()
}
