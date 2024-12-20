#include <stdio.h>
#include <stdlib.h>
#define COM_SIZE 256

// double C = 1, alpha = 0.0001;
double alpha[3] = { 0.0001, 0.0005, 0.001 };

int main() {
    char command[COM_SIZE];
    // sprintf(command, "../libsvm*/svm-train -s 0 -t 2 -c %lf "
    //     "full/full_svm_alpha_%g_mean full/model/full_model_C_%g_alpha_%g_mean_final",
    //     C, alpha, C, alpha);
    // system(command);

    for (int a = 0; a < 3; a++) {
        for (double C = 1e-2; C <= 100; C *= 10) {
            sprintf(command, "../libsvm*/svm-predict "
                "full/2024_full_svm_test full/model/full_model_C_%g_alpha_%g full/predict/full_output_gamma_C_%g_alpha_%g_2024_final",
                C, alpha[a], C, alpha[a]);
            system(command);
        }
    }

    double C = 1;
    sprintf(command, "../libsvm*/svm-predict "
        "full/2024_full_svm_test_mean full/model/full_model_C_%g_alpha_%g_mean_final full/predict/full_output_gamma_C_%g_alpha_%g_2024_mean_final",
        C, alpha[0], C, alpha[0]);
    system(command);

    return 0;
}