#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BUF_SIZE 7000

int main() {
    FILE* fp = fopen("2024_test_data_libsvm", "r");
    FILE* out = fopen("full/2024_full_svm_test", "w");
    char buffer[BUF_SIZE];

    while (fgets(buffer, BUF_SIZE, fp)) {
        char* del = strchr(buffer, '|');
        fprintf(out, "%s", del + 1);
    }

    fclose(fp);
    fclose(out);

    return 0;
}