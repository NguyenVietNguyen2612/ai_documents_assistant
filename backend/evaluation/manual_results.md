# Manual Evaluation Results

## Q1

Question:
What is the main problem addressed in this document?

Expected Answer:
The document addresses the problem of generalizable Deepfake detection and proposes an adaptive frequency-based improvement to FreqNet.

Actual Answer:
The documents address the problem of generalizable Deepfake detection, specifically targeting issues in the original FreqNet architecture where hard masking causes the complete loss of low-frequency semantic information and leads to instability or reduced accuracy when dealing with real-world noisy, blurred, or heavily compressed images. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS


## Q2

Question:
What is FreqNet and what problem is it designed to solve?

Expected Answer:
FreqNet, or Frequency Space Learning Network, is a deep learning architecture designed specifically for generalizable Deepfake detection. It focuses on frequency-domain information rather than relying primarily on pixel-space artifacts.

Actual Answer:
FreqNet, which stands for Frequency Space Learning Network, is an advanced deep learning network architecture. It is specifically designed to solve the problem of generalizable Deepfake detection. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q3

Question:
What are the three main processing blocks of FreqNet?

Expected Answer:
The three main blocks are HFRI, HFRF, and FCL.

Actual Answer:
The three specialized processing blocks that make up the architecture of FreqNet are HFRI, HFRF, and FCL. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q4

Question:
What is the main technical limitation of the original FreqNet identified in the document?

Expected Answer:
The original FreqNet uses hard masking that completely removes the low-frequency components, which can discard important semantic information such as global structure and context.

Actual Answer:
The main technical limitation of the original FreqNet is its hard masking mechanism in the Bh filter, which completely removes the low-frequency band by setting absolute zero values for all components where $|i| < W/4$ and $|j| < H/4$. This causes the model to lose important semantic information, increasing the risk of instability or reduced accuracy when dealing with real-world image variants that are heavily compressed, blurred, or have altered lighting conditions. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q5

Question:
How does Adaptive Soft Masking address the limitation of hard masking in FreqNet?

Expected Answer:
Instead of completely removing the low-frequency components, Adaptive Soft Masking applies learnable weights to the low-frequency region while preserving the high-frequency components with a weight of 1.

Actual Answer:
Adaptive Soft Masking addresses the limitation of hard masking by keeping the high-frequency band intact while replacing the rigid low-frequency cut with a learnable weight matrix regulated by a Sigmoid activation function. This approach preserves essential semantic and structural information instead of completely zeroing it out, preventing network instability. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q6

Question:
How is the adaptive frequency mask defined for low- and high-frequency regions?

Expected Answer:
For the low-frequency region, the adaptive mask is defined as sigma(Wm(i,j)), while for the high-frequency region it is set to 1.

Actual Answer:
The adaptive frequency mask $M_{\text{adapt}}(i, j)$ is defined as $\sigma(W_m(i, j))$ for the low-frequency region where $|i| < \frac{W}{4}$ and $|j| < \frac{H}{4}$, and as $1$ for the high-frequency region in all other cases. Here, $\sigma(\cdot)$ is the Sigmoid activation function and $W_m$ is a learnable weight matrix. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q7

Question:
How is the filtered frequency spectrum calculated in the proposed method?

Expected Answer:
The filtered spectrum is calculated using element-wise Hadamard multiplication: F_tilde = M_adapt element-wise multiplied by F.

Actual Answer:
The proposed method calculates the learned spectrum by using independent convolution layers to compute directly on the amplitude spectrum and the phase spectrum, represented as $\tilde{f}_{am} = L_{conv}(f_{am})$ and $\tilde{f}_{ph} = L_{conv}(f_{ph})$. The new feature matrix is then reconstructed via the inverse Fast Fourier Transform from the sum of these optimized spectra. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q8

Question:
What dataset was used in the experiments, and how was it divided?

Expected Answer:
The experiments used the AI-Generated-vs-Real-Dataset. From the original 152,710 images, the group randomly selected 50,000 images: 36,000 for training, 4,000 for validation, and 10,000 for testing.

Actual Answer:
The dataset used in the experiments was AI-Generated-vs-Real-Dataset, from which 50,000 images were randomly extracted and divided into 36,000 images for training, 4,000 images for validation, and 10,000 images for testing. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q9

Question:
How does the proposed method compare with the original FreqNet in terms of parameters and accuracy?

Expected Answer:
Both models have approximately 1.85 million parameters. FreqNet achieves 72.16% accuracy, while the proposed method achieves 74.03%, corresponding to an improvement of 1.87 percentage points.

Actual Answer:

Both the proposed model and the FreqNet baseline maintain an equivalent number of parameters at approximately 1.85 million. However, the proposed method achieves an improved classification accuracy of 74.03 percent compared to 72.16 percent for FreqNet, representing a 1.87 percent increase in accuracy. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

## Q10

Question:
Was weight decay used to train the model?

Expected Answer:
The document does not specify the weight decay used for training.

Actual Answer:
The provided documents do not contain enough information to answer the question. Source: CS338.pdf, Page 1

Correctness:
PASS

Relevance:
PASS

Groundedness:
PASS

