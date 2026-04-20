
# Fingerprint-Based Blood Group Detection

This repository contains code, datasets, and results related to fingerprint-based blood group detection. The project explores using various machine learning models (ResNet, VGG16, AlexNet, and LeNet) to classify blood groups based on fingerprint images. 

The project is organized into various folders that store datasets, models, performance metrics, and visualization graphs.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Sample Dataset](#sample-dataset)
- [Usage](#usage)

## Folder Structure

The main directories are organized as follows:

```plaintext
Fingerprint-Based-Blood-Group-Detection/
├── code/
│   ├── resnet/          # ResNet model code and files
│   ├── vgg16/           # VGG16 model code and files
│   ├── alexnet/         # AlexNet model code and files
│   └── lenet/           # LeNet model code and files
├── dataset/
│   └── dataset_blood_group/  # Main dataset with folders for each blood group type (A+, B-, etc.)
├── graphs/              # Accuracy, validation accuracy, loss graphs for each model
├── performance_metrix/  # Comparison performance metrics bar graphs
├── sample dataset/      # Sample dataset images
└── test/                # Testing images
```

### Folder Details

- **code/**: Contains subfolders for each model (ResNet, VGG16, AlexNet, LeNet), with Jupyter notebooks files for training and evaluation.
- **dataset/**: Contains the main dataset with images of fingerprints categorized by blood group types (A+, A-, B+, B-, AB+, AB-, O+, O-).
- **graphs/**: Stores accuracy, validation accuracy, loss, and validation loss graphs for each model.
- **performance_metrix/**: Contains bar graphs comparing the performance metrics across different models.
- **sample dataset/**: Includes a sample fingerprint image to preview the dataset.
- **test/**: Placeholder for any testing or additional data as needed.

## Getting Started

### Prerequisites

- Python 3.8 or later
- Jupyter Notebook
- PyTorch
- Additional libraries listed in `requirements.txt`

To install required libraries, run:

```bash
pip install -r requirements.txt
```

### Cloning the Repository

```bash
git clone https://github.com/yourusername/Fingerprint-Based-Blood-Group-Detection.git
cd Fingerprint-Based-Blood-Group-Detection
```

## Dataset

The dataset contains fingerprint images organized into folders by blood group type. There are approximately 6,000–7,000 images spread across the eight blood group categories.

- Each subfolder within `dataset/datasetbloodgroup/` represents a blood group and contains fingerprint images labeled accordingly.
  
Ensure that you have sufficient storage space before downloading or expanding the dataset.

## Models

This project explores the following CNN architectures:

- **ResNet**
- **VGG16**
- **AlexNet**
- **LeNet**

Each model has a dedicated folder under `code/`, with:
- A Jupyter notebook (`.ipynb`) that contains code for training and testing.

## Results

Performance metrics and training graphs are stored in separate folders for easy access and visualization.

- **graphs/**: Includes accuracy, validation accuracy, loss, and validation loss plots for each model, which help visualize the model's performance over time.
- **performance_metrix/**: Contains bar graphs comparing the performance of different models.

## Sample Dataset

The `sample dataset/` folder includes a sample fingerprint image to give users an idea of the dataset's format.

## Usage

1. **Dataset Preparation**:
   Ensure that the `dataset/` folder is populated with the required fingerprint images.

2. **Running the Models**:
   Navigate to the respective model folder in `code/`, open the Jupyter notebook, and run it step-by-step. Each notebook is designed to:
   - Load the dataset
   - Train the model on the training dataset
   - Evaluate the model on the test dataset

3. **Viewing Results**:
   After training, check the `graphs/` and `performance_metrix/` folders for visualizations of the training process and model comparisons.

4. **Testing on Sample Data**:
   You can test the model using images from the `sample dataset/` or by adding additional test images to the `test/` folder.

## Contributing

Feel free to fork this repository, make improvements, and submit a pull request. Contributions to improve model performance, add additional blood group types, or optimize the codebase are welcome.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

Special thanks to the authors of the research paper that served as the foundation for this work. For details on the methodology and findings, refer to the paper included in this repository.
