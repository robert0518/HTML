#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#define COM_SIZE 256
#define DATA_NUM 11067
#define VALID_NUM 8000
#define EXP_NUM 1

// bool valid[DATA_NUM];
double alpha[3] = { 0.0001, 0.0005, 0.001 };

int rangedRandom(int l, int r) {
    return (int)((double)rand() / RAND_MAX * (r - l)) + l;
}

int main() {
    for (int exp = 0; exp < EXP_NUM; exp++) {
        // srand(exp);

        // memset(valid, 0, sizeof(valid));
        // for (int i = 0; i < VALID_NUM; i++) {
        //     int temp = rangedRandom(0, DATA_NUM);
        //     while (valid[temp])
        //         temp = rangedRandom(0, DATA_NUM);
        //     valid[temp] = 1;
        // }

        for (int a = 0; a < 3; a++) {
            // char filename[100];
            // sprintf(filename, "full/full_svm_alpha_%g", alpha[a]);
            // FILE* fp = fopen(filename, "r");
            // sprintf(filename, "full/train_alpha_%g", alpha[a]);
            // FILE* trainfile = fopen(filename, "w");
            // sprintf(filename, "full/valid_alpha_%g", alpha[a]);
            // FILE* validfile = fopen(filename, "w");

            // char buffer[7000];
            // for (int i = 0; i < DATA_NUM; i++) {
            //     fgets(buffer, sizeof(buffer), fp);
            //     if (valid[i]) {
            //         fprintf(trainfile, "%s", buffer);
            //     } else {
            //         fprintf(validfile, "%s", buffer);
            //     }
            // }
            // fclose(fp);
            // fclose(trainfile);
            // fclose(validfile);

            char command[COM_SIZE];

            for (double C = 1e-2; C <= 100; C *= 10) {
                // for (double gamma = 1e-4; gamma <= 100; gamma *= 10) {
                printf("exp:%d C:%g alpha:%g\n", exp, C, alpha[a]);
                sprintf(command, "../libsvm*/svm-train -q -s 0 -t 2 -c %lf -v 10 "
                    "full/full_svm_alpha_%g_mean full/model/full_model_C_%g_alpha_%g_mean",
                    C, alpha[a], C, alpha[a]);
                system(command);
                // }
            }

            // for (double C = 1e-2; C <= 100; C *= 10) {
            //     // for (double gamma = 1e-4; gamma <= 100; gamma *= 10) {
            //     printf("exp:%d C:%g alpha:%g\n", exp, C, alpha[a]);
            //     sprintf(command, "../libsvm*/svm-predict "
            //         "full/valid_alpha_%g full/model/full_model_C_%g_alpha_%g_mean full/predict/full_output_C_%g_alpha_%g_mean",
            //         alpha[a], C, alpha[a], C, alpha[a]);
            //     system(command);
            //     // }
            // }
        }
    }

    return 0;
}