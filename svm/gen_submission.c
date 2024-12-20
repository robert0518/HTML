#include <stdio.h>
#include <string.h>

int main() {
    FILE* fp = fopen("full/predict/full_output_gamma_C_1_alpha_0.0001_2024_mean_final", "r");
    FILE* out = fopen("full_svm_2024_mean_submission.csv", "w");
    char buffer[10];

    fprintf(out, "id,home_team_win\n");
    int i = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        *strchr(buffer, '\n') = 0;
        fprintf(out, "%d,%s\n", i++, (buffer[0] == '1') ? "True" : "False");
    }

    fclose(fp);
    fclose(out);

    return 0;
}