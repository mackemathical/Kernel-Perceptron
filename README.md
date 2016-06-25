# Kernel-Perceptron
This is a dual perceptron, which normally uses dot products between data points, allowing a kernel function to be used, generalizing the Perceptron algorithm to non-linear problems.

Kernel_Perceptron.py contains the entire program, the other python files are simply ready-to-go tests on the algorithm. KPIris is a test on popular the Iris data from http://archive.ics.uci.edu/ml/datasets/Iris. KPXor attempts to learn the Xor function from some artificial. KPCrescent attempts to learn a more generic non-linearly seperable classification. In all cases (which are cherry-picked to showcase the kernel trick) the algorithm performs extraordinarily well; and better results are possible by tweaking perameters (not explored here, but you're encouraged to download the code and play with it!).

If you have a classification task which an SVM can do better than my dual perceptron, please contact me through github (@mackemathical)! I'd really like to play with something harder to explore this algorithm further.
