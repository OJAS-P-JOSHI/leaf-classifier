# Leaf Disease Classifier: Identify Potato and Mango Leaf Diseases with Deep Learning

[![Screenshot-2024-05-03-191041.png](https://i.postimg.cc/1tJxVL3q/Screenshot-2024-05-03-191041.png)](https://postimg.cc/KkkpdpLG)

[![Screenshot-2024-05-03-191915.png](https://i.postimg.cc/RZBz81yq/Screenshot-2024-05-03-191915.png)](https://postimg.cc/XpQhB5V0)

[![Screenshot-2024-05-03-191813.png](https://i.postimg.cc/W3QyZTPt/Screenshot-2024-05-03-191813.png)](https://postimg.cc/k6vw3rGd)

This project is a web application built using Streamlit that empowers users to classify leaf diseases in potato and mango plants. By leveraging the power of deep learning, it offers a user-friendly interface for image upload and disease prediction, making plant health monitoring a breeze.

## Key Features

* **Accurate Disease Classification:**
    * Classifies potato leaf diseases into: Early Blight, Late Blight, Healthy.
    * Classifies mango leaf diseases into: Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Healthy, Powdery Mildew, Sooty Mould.
* **Comprehensive User Guide:**
    * Access a clear video guide that demonstrates effective application usage.
* **In-Depth Disease Information:**
    * Gain valuable insights about potato and mango leaf diseases to make informed decisions about plant care.

## Prerequisites

- Python 3.7 or later
- Streamlit (`pip install streamlit`)
- TensorFlow (`pip install tensorflow`)
- Keras (`pip install keras`)
- Pillow (PIL Fork) (`pip install Pillow`)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/OJAS-P-JOSHI/leaf-classifier](https://github.com/OJAS-P-JOSHI/leaf-classifier)
   ```


2. **Install Dependencies:**

Navigate to the project directory and run:
```bash
cd leaf-disease-classifier
pip install -r requirements.txt
```
## Usage
A) Start the application:

```bash
streamlit run Main.py
 ```

B) The web app will open in your default browser.

C) Upload an image of a potato or mango leaf.

D) The application will classify the disease and display the results.

## Datasets

* **Mango Leaf Disease Dataset:** Used to train the Mango Leaf Disease Classifier model. It contains images of mango leaves with various diseases. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset/data).

* **Potato Leaf Disease Dataset:** Used to train the Potato Leaf Disease Classifier model. It contains images of potato leaves with various diseases. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village). This dataset includes images of various plant diseases, so you'll need to filter or select the potato-specific images for training your model.

## Contributing

We welcome your valuable contributions! If you have suggestions for improvements or want to fix issues, feel free to open an issue or create a pull request on GitHub.

## License

This project is licensed under the MIT License.
