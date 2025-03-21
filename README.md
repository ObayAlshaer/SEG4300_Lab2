# House Segmentation Pipeline

**Mohamed-Obay Alshaer**  
**300170489**  
**SEG4300**  
**March 21, 2025**

This repository contains an enhanced version of the Lab 1 pipeline, which originally served a sentiment analysis model through a Flask API. The enhancements include:

1. **Secrets Management**: Added secure handling of sensitive information like API keys and credentials
2. **CI/CD Implementation**: Set up automated testing, building, and deployment
3. **Model Replacement**: Replaced the sentiment analysis model with a segmentation model trained on aerial footage for house segmentation

## Project Structure

```
house-segmentation-pipeline/
│
├── .github/workflows/   # CI/CD configuration
├── app/                 # Flask application code
├── models/              # Model checkpoints
├── notebook/            # Jupyter notebook with data preparation, training, and evaluation code
├── tests/               # Automated tests
├── .env.example         # Template for environment variables
├── Dockerfile           # Docker configuration
├── entrypoint.sh        # Docker entrypoint script
└── requirements.txt     # Project dependencies
```

## Features

### 1. Secrets Management

The application now securely manages sensitive information:

- Environment variables loaded from `.env` file using python-dotenv
- Docker entrypoint script for secure secrets injection
- API key authentication for endpoints

Example of the `.env` file structure:
```
API_KEY=your_secret_api_key_here
MODEL_PATH=./models/segmentation_model.pth
DEBUG_MODE=False
```

### 2. CI/CD Pipeline

Automated testing and deployment workflow using GitHub Actions:

- **Testing Stage**: Unit tests for API and model functionality
- **Build Stage**: Docker image building and tagging
- **Deploy Stage**: Pushing to Docker Hub and deployment

The pipeline ensures code quality and consistent deployments.

### 3. House Segmentation Model

A UNet architecture trained on aerial footage for house segmentation:

- Dataset preparation using pixel mask generation
- Model training with proper metrics tracking (IoU, Dice score)
- Evaluation and visualization of segmentation results

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Docker (for containerized deployment)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/house-segmentation-pipeline.git
   cd house-segmentation-pipeline
   ```

2. **Set up environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure secrets:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Jupyter Notebook

The Jupyter notebook contains the complete pipeline for dataset preparation, model training, and evaluation:

```bash
cd notebook
jupyter notebook "House Segmentation Pipeline.ipynb"
```

### Running the Flask API

1. **Start the Flask application:**
   ```bash
   python app/app.py
   ```

2. **Make predictions:**
   ```bash
   curl -X POST http://localhost:5001/predict \
        -H "X-API-Key: your_api_key" \
        -F "image=@/path/to/image.jpg"
   ```

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t house-segmentation .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5001:5001 --env-file .env house-segmentation
   ```

## Model Performance

The house segmentation model achieves:

- **IoU Score**: 0.85 (average on test set)
- **Dice Score**: 0.90 (average on test set)

Visualizations of sample predictions and metrics distributions are available in the notebook and evaluation results.

## References

1. Original Lab 1 sentiment analysis pipeline
2. Week 7 pixel mask generation code
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention*.
4. "keremberke/satellite-building-segmentation" dataset from Hugging Face.

## License

This project is licensed under the MIT License - see the LICENSE file for details.