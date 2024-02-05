# Treatment of Essential Tremor through Machine Learning and Deep Learning

## Overview
Essential tremor is a neurological disorder that causes the appearance of pathological tremor in parts of the body or limbs, potentially compromising the patient's ability to perform daily activities. Currently, there is no effective and accessible treatment for all patients. This project explores the stimulation of afferent pathways, combined with an out-of-phase stimulation strategy, as an economical and non-invasive alternative to overcome the limitations of conventional treatments.

## Introduction
Recent research suggests that the out-of-phase strategy, which involves applying stimulations to pairs of antagonist muscles synchronized with the muscle activity phase of the tremor, could be a promising solution. However, predicting tremor cycles based on their frequency presents challenges due to the non-stationary nature of tremor signals. This project proposes the use of traditional machine learning algorithms and an LSTM neural network to improve the classification and prediction of tremor cycles.

## Objectives
- To explore and validate the use of traditional Machine Learning algorithms (KNN, SVM, and Random Forest) and LSTM neural networks for the classification and prediction of tremor kinematic signals.
- To implement an optimized out-of-phase stimulation strategy for the suppression of pathological tremor.

## Methodology
To evaluate the classification and prediction algorithms, two datasets were created with kinematic signals corresponding to the temporal variations of wrist flexo-extension angles. The classification dataset included data from 12 patients with essential tremor and 5 healthy subjects, resulting in a total of 18,000 labeled segments. The prediction dataset consisted of 7,000 temporal segments from patients with essential tremor.

## Results
The implemented classifiers achieved accuracies over 90% in the binary classification task of tremor and non-tremor kinematic signals. The LSTM neural network proved capable of robustly and satisfactorily predicting future tremor cycles with a prediction horizon of up to 1 second. Correlation values for the first 100 ms were above 0.9, decreasing to 0.75 for the 1-second future prediction.

## Conclusion
The results confirm that the use of machine learning and deep learning algorithms can significantly optimize the out-of-phase strategy in the stimulation of afferent pathways, achieving greater suppression of pathological tremor. This approach represents a promising advancement in the treatment of essential tremor, offering a more accessible and effective solution for patients.

## How to Contribute
This project is open to contributions from the scientific and medical community. If you are interested in collaborating or implementing this technology, please contact the authors.

## License
This project is distributed under the [insert license type] license, which allows use, modification, and distribution under certain conditions.
